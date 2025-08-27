import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mistral.modeling_mistral import MistralMLP

class MoELayer(nn.Module):
    def __init__(self, d_model: int, num_experts: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts

        # The router is a simple linear layer that outputs a score for each expert
        self.router = nn.Linear(d_model, num_experts)

        # The experts are a list of standard Mistral FFNs (MLPs)
        self.experts = nn.ModuleList([MistralMLP() for _ in range(num_experts)])

    def forward(self, hidden_states: torch.Tensor, deterministic_expert_idx: int = None):
        """
        Forward pass for the MoE layer.
        
        Args:
            hidden_states (torch.Tensor): The input tensor from the previous layer.
            deterministic_expert_idx (int, optional): If provided, forces all tokens
                                                     to be routed to this expert.
                                                     Used for Stage 2 training.
        """
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, d_model) # (batch*seq, d_model)

        if deterministic_expert_idx is not None:
            # Stage 2: Hard-coded routing
            expert_output = self.experts[deterministic_expert_idx](hidden_states)
            return expert_output
        else:
            # Stage 3: Router-based gating
            router_logits = self.router(hidden_states_flat) # (batch*seq, num_experts)
            
            # Use top-1 gating
            routing_weights = F.softmax(router_logits, dim=1)
            top_k_weights, top_k_indices = torch.topk(routing_weights, 1, dim=1)
            
            # The final output will be a weighted sum (in this case, just the top expert)
            final_hidden_states = torch.zeros_like(hidden_states_flat)
            
            # This loop can be optimized, but is clear for demonstration
            for i in range(self.num_experts):
                # Find which tokens are routed to this expert
                mask = (top_k_indices == i).squeeze(1)
                
                if mask.any():
                    # Process the selected tokens with the corresponding expert
                    expert_input = hidden_states_flat[mask]
                    expert_output = self.experts[i](expert_input.unsqueeze(1)).squeeze(1)
                    
                    # Apply gating weights and add to the final output
                    final_hidden_states.masked_scatter_(mask.unsqueeze(1), expert_output * top_k_weights[mask])

            return final_hidden_states.view(batch_size, seq_len, d_model)