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
lr = 1e-4
node_embed_dim = 64
n_mp_phases = 16
n_readout_depth = 8
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
model.compile()
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
    / f"{datetime.now().strftime('%m%d_%H%M%S')}_{lr=}{node_embed_dim=}{n_mp_phases=}{n_readout_depth=}"
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
MAX_EXAMPLES = 10_000
n_epochs = 30
torch.manual_seed(2026)
train_loader, val_loader, test_loader = data.load_joined_data(
    batch_size=BATCH_SIZE,
    max_examples=MAX_EXAMPLES,
    train_fraction=0.7,
    val_fraction=0.15,
    split_seed=42,
)
test_example = next(iter(train_loader))


target_norms = {
    "pbe0_energy": {
        "count": 33496171,
        "mean": -2443.8402999739274,
        "std": 4265.76866428051,
    },
    "pbe0_formation_energy": {
        "count": 33496171,
        "mean": -1.750654284213469,
        "std": 0.8610489154378447,
    },
}
TARGET_NAME = "pbe0_formation_energy"
TARGET_STATS = target_norms[TARGET_NAME]
METRIC_NAMES = ("loss", "rel_err", "mae", "rmse")
ACCUM_METRIC_NAMES = ("loss", "rel_err", "mae", "energy_mse")


def _energy_metrics(pred, target_energy):
    pred_energy = pred * TARGET_STATS["std"] + TARGET_STATS["mean"]
    mae = torch.mean(torch.abs(pred_energy - target_energy))
    energy_mse = torch.mean((pred_energy - target_energy) ** 2)
    return mae, energy_mse


def _prepare_batch(example):
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
    target_energy = example[TARGET_NAME].to(device)
    target = (target_energy - TARGET_STATS["mean"]) / TARGET_STATS["std"]
    return (
        atoms.to(device),
        positions.to(device),
        mol_graphs.to(device),
        target,
        target_energy,
    )


def _compute_metrics(pred, target, target_energy, loss):
    rel_err = torch.mean(torch.abs(pred - target) / target.abs().clamp_min(1e-12))
    mae, energy_mse = _energy_metrics(pred, target_energy)
    return {
        "loss": loss.item(),
        "rel_err": rel_err.item(),
        "mae": mae.item(),
        "energy_mse": energy_mse.item(),
    }


def _format_metrics(metrics):
    return " ".join(f"{name}={metrics[name]:.6f}" for name in METRIC_NAMES)


def run_epoch(model, loader, *, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    totals = {name: 0.0 for name in ACCUM_METRIC_NAMES}
    n_examples = 0

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for example in loader:
            try:
                atoms, positions, mol_graphs, target, target_energy = _prepare_batch(
                    example
                )
            except Exception as e:
                print("Skipping batch:", e, file=sys.stderr)
                continue

            pred = model(atoms, positions, mol_graphs)
            loss = loss_fn(pred, target)
            batch_size = target.shape[0]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = _compute_metrics(pred, target, target_energy, loss)
            for name, value in metrics.items():
                totals[name] += value * batch_size
            n_examples += batch_size

    if n_examples == 0:
        raise RuntimeError("No batches were processed.")

    metrics = {name: totals[name] / n_examples for name in ACCUM_METRIC_NAMES}
    metrics["rmse"] = metrics.pop("energy_mse") ** 0.5
    return metrics


for epoch in range(n_epochs):
    train_metrics = run_epoch(model, train_loader, optimizer=optim)
    val_metrics = run_epoch(model, val_loader)

    _add_tensorboard_logs(logger, train_metrics, epoch, mode="train")
    _add_tensorboard_logs(logger, val_metrics, epoch, mode="val")

    print(
        f"epoch={epoch + 1}/{n_epochs}",
        f"train {_format_metrics(train_metrics)}",
        f"val {_format_metrics(val_metrics)}",
    )

test_metrics = run_epoch(model, test_loader)
_add_tensorboard_logs(logger, test_metrics, n_epochs, mode="test")
print(f"test {_format_metrics(test_metrics)}")
