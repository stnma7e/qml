import math

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    # TODO: learn about / add dropout
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        Q = self.Wq(q)
        K = self.Wk(k)
        attn = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.embed_dim), dim=-1)
        return attn @ self.Wv(v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = Attention(embed_dim)
        self.mlp = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_mask):
        return x + x_mask * F.relu(
            self.mlp(
                self.attn(x, x, x),
            ),
        )


class EnergyNet(nn.Module):
    def __init__(self, n_atom_max: int, embed_dim: int, n_layers: int):
        super().__init__()
        self.n_atom_max = n_atom_max
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(4 * n_atom_max, embed_dim)
        self.transformers = nn.ModuleList(
            [TransformerLayer(self.embed_dim) for _ in range(n_layers)]
        )
        self.energy_mlp = nn.Linear(embed_dim, 1)

    def forward(self, atoms, positions, atom_mask):
        padding = torch.zeros(atoms.shape[0], self.n_atom_max - atoms.shape[1]).to(
            atoms.device
        )
        atoms = torch.cat([atoms, padding], dim=1)
        positions = torch.cat(
            [positions, padding.unsqueeze(2).expand(-1, -1, 3)],
            dim=1,
        )
        x = torch.cat([atoms.unsqueeze(2), positions], dim=2)
        x = self.embedding(x.view(x.shape[0], -1))
        for layer in self.transformers:
            x = layer(x, atom_mask)
        return self.energy_mlp(x)
