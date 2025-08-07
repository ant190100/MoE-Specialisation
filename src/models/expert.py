"""
Expert module for MoE models.
"""

import torch
import torch.nn as nn


class Expert(nn.Module):
    """A simple feed-forward network to be used as an expert."""

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),  # Project back to the original dimension
        )

    def forward(self, x):
        return self.net(x)
