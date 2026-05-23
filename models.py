import math

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    # TODO: learn about / add dropout
    def __init__(self, embed_dim: int, p_dropout: float = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.p_dropout = p_dropout

    def forward(self, q, k, v, mask):
        Q = self.Wq(q)
        K = self.Wk(k)
        attn = Q @ K.transpose(-2, -1) / math.sqrt(self.embed_dim)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.p_dropout)
        return attn @ self.Wv(v)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = Attention(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            *[
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.ReLU(),
                nn.Linear(4 * embed_dim, embed_dim),
            ]
        )

    def forward(self, x, mask):
        x = self.ln1(x)
        x = x + self.attn(x, x, x, mask)
        return self.mlp(self.ln2(x))


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
        x = torch.cat([atoms.unsqueeze(2), positions], dim=2)
        x = self.embedding(x.view(x.shape[0], -1))
        for layer in self.transformers:
            x = layer(x, atom_mask)
        return self.energy_mlp(x)


class MolGraph(nn.Module):
    def __init__(
        self,
        mol_size: int,
        node_embed_dim: int,
        edge_embed_dim: int,
        n_mp_phases: int,
        n_readout_depth: int,
        n_atom_classes: int = 130,
    ):
        super().__init__()

        self.mol_size = mol_size
        self.node_embed_dim = node_embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.n_mp_phases = n_mp_phases

        self.node_embed = nn.Embedding(
            n_atom_classes,
            self.node_embed_dim,
        )

        # this is not the MPNN M function
        # theirs learns a matrix A(e_vv) and multiplies against the node embedding h
        self.M = nn.Sequential(
            *[
                nn.Linear(
                    self.node_embed_dim + self.edge_embed_dim,
                    4 * self.node_embed_dim + self.edge_embed_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    4 * self.node_embed_dim + self.edge_embed_dim,
                    self.node_embed_dim + self.edge_embed_dim,
                ),
            ]
        )
        self.U = nn.Sequential(
            *[
                nn.Linear(
                    self.node_embed_dim + self.edge_embed_dim,
                    4 * self.node_embed_dim + self.edge_embed_dim,
                ),
                nn.ReLU(),
                nn.Linear(
                    4 * self.node_embed_dim + self.edge_embed_dim,
                    self.node_embed_dim,
                ),
            ]
        )
        self.R_transformers = nn.ModuleList(
            [TransformerLayer(self.node_embed_dim) for _ in range(n_readout_depth)]
        )
        self.R_linear = nn.Linear(self.mol_size * self.node_embed_dim, 1)

    def forward(self, atoms, edges, mol_graphs):
        atom_embeddings = self.node_embed(atoms)
        # TODO decide if I can do this all as a single sparse matrix op, or if I need to break it into a loop over active edges
        for phase in range(self.n_mp_phases):
            for atom in range(atom_embeddings.shape[1]):
                messages = torch.zeros(
                    (
                        atoms.shape[0],
                        self.node_embed_dim + self.edge_embed_dim,
                    )
                ).to(atoms.device)
                for other in range(atom_embeddings.shape[1]):
                    messages += self.M(
                        torch.cat(
                            (
                                atom_embeddings[:, other],
                                edges[:, atom, other],
                            ),
                            dim=1,
                        ),
                    )
                atom_embeddings[:, atom] += self.U(messages)

        x = atom_embeddings
        for layer in self.R_transformers:
            x = layer(x, mol_graphs)
        return self.R_linear(x.view(x.shape[0], -1)).squeeze()
