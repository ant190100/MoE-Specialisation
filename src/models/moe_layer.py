"""
Mixture of Experts layer implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .expert import Expert


class MoELayer(nn.Module):
    """
    The core Mixture-of-Experts layer.
    This layer takes a batch of tokens and routes each token to the top-k experts.
    """

    def __init__(self, embed_dim, hidden_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Create the pool of experts
        self.experts = nn.ModuleList(
            [Expert(embed_dim, hidden_dim) for _ in range(num_experts)]
        )

        # The gating network (router) is a simple linear layer that outputs
        # a logit for each expert.
        self.router = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)
        batch_size, seq_len, embed_dim = x.shape

        # Reshape the input to be a flat list of tokens for easier processing
        x_flat = x.view(-1, embed_dim)  # Shape: (batch_size * seq_len, embed_dim)
        num_tokens = x_flat.shape[0]

        # 1. Gating / Routing
        # Get the logits from the router for each token
        router_logits = self.router(x_flat)  # Shape: (num_tokens, num_experts)

        # Find the top-k experts and their corresponding routing weights
        routing_weights, chosen_expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = F.softmax(
            routing_weights, dim=-1
        )  # Softmax over the top-k logits

        # 2. Expert Processing
        # Initialize the final output tensor
        final_output = torch.zeros_like(x_flat)

        # Create a flat index to map tokens to their chosen experts
        flat_expert_indices = chosen_expert_indices.view(-1)

        # Create a tensor that maps each token to its position in the batch
        token_batch_map = torch.arange(num_tokens, device=x.device).repeat_interleave(
            self.top_k
        )

        # Get the expert outputs for all tokens and all chosen experts
        # This is a more complex but efficient way to handle batching for top-k
        expert_outputs = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            # Find which tokens have this expert in their top-k list
            mask = (chosen_expert_indices == i).any(dim=-1)
            if mask.any():
                expert_outputs[mask] = self.experts[i](x_flat[mask])

        # Combine the expert outputs using the routing weights
        # We need to gather the correct expert outputs for each token
        # and multiply by the corresponding routing weight.
        weighted_outputs = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = chosen_expert_indices[:, i]
            weight = routing_weights[:, i].unsqueeze(1)

            # Gather the outputs from the correct experts
            # This is a bit complex, but it avoids a slow loop
            current_expert_outputs = torch.zeros_like(x_flat)
            for j in range(self.num_experts):
                mask = expert_idx == j
                if mask.any():
                    current_expert_outputs[mask] = self.experts[j](x_flat[mask])

            weighted_outputs += weight * current_expert_outputs

        # Reshape the output back to the original input shape
        return weighted_outputs.view(batch_size, seq_len, embed_dim)
