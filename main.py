# %%
# %load_ext autoreload
# %autoreload 2
# %capture --no-stderr --no-display

# %%

import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F

import data
import models

device = torch.device("mps")

N_ATOM_MAX = 32
model = models.EnergyNet(
    n_atom_max=N_ATOM_MAX,
    embed_dim=64,
    n_layers=4,
).to(device)
model = models.MolGraph(
    mol_size=N_ATOM_MAX,
    node_embed_dim=64,
    edge_embed_dim=64,
    n_readout_depth=2,
).to(device)
optim = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
)
loss_fn = nn.MSELoss()

print(model)

BOND_DISTANCE_BUCKETS = torch.cat(
    (torch.tensor([0]), torch.linspace(2, 6, 8), torch.tensor([torch.inf])), dim=0
)


def _bucket_atom_positions(batch_positions):
    dist = torch.cdist(batch_positions, batch_positions)
    buckets = torch.bucketize(dist, BOND_DISTANCE_BUCKETS)
    return F.one_hot(buckets)


BATCH_SIZE = 256
n_epochs = 1
torch.manual_seed(2026)
dataloader = data.load_joined_data(batch_size=BATCH_SIZE)

batch = next(iter(dataloader))

for epoch in range(n_epochs):
    model.train()
    # for i, example in enumerate(dataloader):
    for i, example in [(i, batch) for i in range(int(1e4))]:
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
                for graph in [
                    torch.tensor(Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(m)))
                ]
            ]
        )
        # print(atoms.shape)
        # print(positions.shape)
        # print(mol_graphs)
        pred = model(atoms.to(device), positions.to(device), mol_graphs.to(device))
        target = torch.tensor(example["pbe0_energy"]).to(device)
        # print(pred.shape, target.shape)
        loss = loss_fn(pred, target)
        optim.zero_grad()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None:
                print("no grad:", name)
            elif not torch.isfinite(p.grad).all():
                print("bad grad:", name)
            elif p.grad.abs().mean() == 0:
                print("zero-ish grad:", name)

        optim.step()

        with torch.no_grad():
            if i % 100 == 0:
                relative_acc = torch.mean(torch.abs(pred - target) / target)
                print(
                    relative_acc,
                    loss / BATCH_SIZE,
                )
