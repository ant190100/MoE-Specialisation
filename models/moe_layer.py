import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mistral.modeling_mistral import MistralMLP

class MoELayer(nn.Module):
    def __init__(self, config, d_model: int, num_experts: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([MistralMLP(config) for _ in range(num_experts)])

    def forward(self, hidden_states: torch.Tensor):
        routing_mask = self.routing_mask
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Initialize output
        final_output = torch.zeros_like(hidden_states)

        # Route to vision expert (0) - ONLY process vision tokens
        vision_mask = (routing_mask == 0)
        if vision_mask.any():
            # Extract ONLY the vision tokens (memory efficient)
            vision_indices = vision_mask.nonzero(as_tuple=True)
            vision_tokens = hidden_states[vision_indices]  # Only vision tokens
            
            if vision_tokens.numel() > 0:
                vision_output = self.experts[0](vision_tokens)
                final_output[vision_indices] = vision_output

        # Route to text expert (1) - ONLY process text tokens  
        text_mask = (routing_mask == 1)
        if text_mask.any():
            # Extract ONLY the text tokens (memory efficient)
            text_indices = text_mask.nonzero(as_tuple=True)
            text_tokens = hidden_states[text_indices]  # Only text tokens
            
            if text_tokens.numel() > 0:
                text_output = self.experts[1](text_tokens)
                final_output[text_indices] = text_output

        return final_output