"""
Complete TinyMoE model for classification.
"""

import torch
import torch.nn as nn
from .moe_layer import MoELayer


class TinyMoEForClassification(nn.Module):
    """The main model that uses the MoE layer for classification."""

    def __init__(
        self, vocab_size, embed_dim, hidden_dim, num_experts, top_k, num_classes
    ):
        super().__init__()
        # The embedding layer turns token IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Our custom MoE layer
        self.moe_layer = MoELayer(embed_dim, hidden_dim, num_experts, top_k)

        # A simple linear layer to map the output of the MoE layer to class predictions
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # input_ids shape: (batch_size, sequence_length)
        embedded = self.embedding(input_ids)  # Shape: (batch_size, seq_len, embed_dim)

        moe_output = self.moe_layer(embedded)  # Shape: (batch_size, seq_len, embed_dim)

        # We use the representation of the first token ([CLS] token) for classification.
        # This is a common practice in models like BERT.
        cls_token_output = moe_output[:, 0]  # Shape: (batch_size, embed_dim)

        # Get the final logits for each class
        logits = self.classifier(cls_token_output)  # Shape: (batch_size, num_classes)
        return logits
