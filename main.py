# %%
%load_ext autoreload
%autoreload 2

# %%

import torch
from torch import nn
from torch.nn import functional as F
import data
import models

device = torch.device("mps")

n_epochs=10
torch.manual_seed(2026)
dataloader = data.load_data(
    batch_size=64,
)

N_ATOM_MAX = 64
model = models.EnergyNet(
    n_atom_max=N_ATOM_MAX,
    embed_dim=64,
    n_layers=4,
).to(device)
optim = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
)
loss_fn = nn.MSELoss()

print(model)

BOND_DISTANCE_BUCKETS = torch.cat((torch.tensor([0]), torch.linspace(2, 6, 8), torch.tensor([torch.inf])), dim=0)

def _bucket_atom_positions(batch_positions):
    dist = torch.cdist(batch_positions, batch_positions)
    buckets = torch.bucketize(dist, BOND_DISTANCE_BUCKETS)
    return F.one_hot(buckets)

for epoch in range(n_epochs):
    model.train()
    for i, example in enumerate(dataloader):
        # print(example.keys())
        atom_data =  torch.stack(example["atomic_numbers"])
        positions =  _bucket_atom_positions(torch.stack(example["positions"]))
        positions = torch.stack(example["positions"])
        mask = torch.ones(N_ATOM_MAX)
        pred = model(atom_data.to(device), positions.to(device), mask.to(device))
        # print(pred.shape)
        loss = loss_fn(pred.squeeze(), torch.tensor(example["pbe0_energy"]).to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (i % 10000 == 0):
            print(loss)
