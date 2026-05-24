# %%
# %load_ext autoreload
# %autoreload 2
# %capture --no-stderr --no-display

# %%

import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb
from rdkit import Chem
from torch import nn
from torch.nn import functional as F

import data
import models

# torch.autograd.set_detect_anomaly(True, check_nan=False)
device = torch.device("mps")


def _add_tensorboard_logs(logger: tb.SummaryWriter, metrics, step, mode="train"):
    for k, v in metrics.items():
        logger.add_scalar(f"{mode}/{k}", v, step)


N_ATOM_MAX = 32
lr = 1e-5
node_embed_dim = 64
n_mp_phases = 4
n_readout_depth = 2
model = models.EnergyNet(
    n_atom_max=N_ATOM_MAX,
    embed_dim=64,
    n_layers=4,
).to(device)
model = models.MolGraph(
    mol_size=N_ATOM_MAX,
    node_embed_dim=node_embed_dim,
    edge_embed_dim=10,
    n_mp_phases=n_mp_phases,
    n_readout_depth=n_readout_depth,
).to(device)
optim = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
)
loss_fn = nn.MSELoss()

print(model)
print(sum([p.numel() for p in model.parameters()]))

exp_dir = "logs"
log_dir = (
    Path(exp_dir)
    / f"{datetime.now().strftime('%m%d_%H%M%S')}_{lr=}{node_embed_dim=}{n_mp_phases}{n_readout_depth}"
)
logger = tb.SummaryWriter(log_dir)


BOND_DISTANCE_BUCKETS = torch.cat(
    (torch.tensor([0]), torch.linspace(2, 6, 8), torch.tensor([torch.inf])), dim=0
)


def _bucket_atom_positions(batch_positions):
    dist = torch.cdist(batch_positions, batch_positions)
    buckets = torch.bucketize(dist, BOND_DISTANCE_BUCKETS)
    return F.one_hot(buckets)


BATCH_SIZE = 64
MAX_EXAMPLES = 100_000
n_epochs = 100
torch.manual_seed(2026)
train_loader, val_loader, test_loader = data.load_joined_data(
    batch_size=BATCH_SIZE,
    max_examples=MAX_EXAMPLES,
    train_fraction=0.7,
    val_fraction=0.15,
    split_seed=42,
)


target_norms = {
    "pbe0_energy": {
        "count": 33496171,
        "mean": -2443.8402999739274,
        "std": 4265.76866428051,
    }
}


def _transform_example(example):
    # print(example.keys())
    atoms = torch.stack(
        [
            torch.cat((a, torch.zeros(N_ATOM_MAX - a.shape[0], dtype=torch.long)))
            for a in example["atomic_numbers"]
        ]
    )
    positions = torch.stack(
        [
            torch.cat((p, torch.zeros(N_ATOM_MAX - p.shape[0], p.shape[1])))
            for p in example["positions"]
        ]
    )
    positions = _bucket_atom_positions(positions)
    mol_graphs = torch.stack(
        [
            torch.eye(N_ATOM_MAX, N_ATOM_MAX)
            + F.pad(
                graph,
                (0, N_ATOM_MAX - graph.shape[0], 0, N_ATOM_MAX - graph.shape[0]),
            )
            for m in example["smiles"]
            for graph in [torch.tensor(Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(m)))]
        ]
    )
    return atoms, positions, mol_graphs


test_batch = next(iter(train_loader))
global_step = 0
for epoch in range(n_epochs):
    model.train()
    train_metrics = {"loss": 0.0, "err": 0.0}
    for example in train_loader:
        try:
            atoms, positions, mol_graphs = _transform_example(example)
        except Exception as e:
            print("Skipping batch:", e, file=sys.stderr)
            continue
        # print(atoms.shape)
        # print(positions.shape)
        # print(mol_graphs.shape)
        pred = model(atoms.to(device), positions.to(device), mol_graphs.to(device))
        target = example["pbe0_energy"].to(device)
        target = (target - target_norms["pbe0_energy"]["mean"]) / (
            target_norms["pbe0_energy"]["std"]
        )
        # print(pred.shape, target.shape)
        loss = loss_fn(pred, target)
        optim.zero_grad()
        loss.backward()
        optim.step()

        relative_err = torch.mean(
            torch.abs(pred - target) / target.abs().clamp_min(1e-12)
        )

        train_metrics["loss"] += loss.item()
        train_metrics["err"] += relative_err.item()

        global_step += 1
        if global_step == 1 or global_step % 100 == 0:
            print(
                loss.item(),
                relative_err.item(),
            )

            val_metrics = {"loss": 0.0, "err": 0.0}
            with torch.inference_mode():
                model.eval()
                for example in val_loader:
                    try:
                        atoms, positions, mol_graphs = _transform_example(example)
                    except Exception as e:
                        print("Skipping batch:", e, file=sys.stderr)
                        continue
                    pred = model(
                        atoms.to(device), positions.to(device), mol_graphs.to(device)
                    )
                    target = example["pbe0_energy"].to(device)
                    target = (target - target_norms["pbe0_energy"]["mean"]) / (
                        target_norms["pbe0_energy"]["std"]
                    )
                    loss = loss_fn(pred, target)
                    relative_err = torch.mean(
                        torch.abs(pred - target) / target.abs().clamp_min(1e-12)
                    )

                    val_metrics["loss"] += loss.item()
                    val_metrics["err"] += relative_err.item()

            val_metrics["loss"] /= len(val_loader)
            val_metrics["err"] /= len(val_loader)
            train_metrics["loss"] /= 100
            train_metrics["err"] /= 100
            _add_tensorboard_logs(logger, train_metrics, global_step)
            _add_tensorboard_logs(logger, val_metrics, global_step, mode="val")
            train_metrics = {"loss": 0.0, "err": 0.0}
