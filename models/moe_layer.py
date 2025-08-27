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

    def forward(self, hidden_states: torch.Tensor):
        # The routing mask is passed as an attribute, not an argument
        # This is a workaround for the standard Hugging Face forward signature
        routing_mask = self.routing_mask
        final_output = torch.zeros_like(hidden_states)

        # Route to vision expert (0)
        vision_mask = (routing_mask == 0).unsqueeze(-1)
        if vision_mask.any():
            vision_tokens = torch.where(
                vision_mask,
                hidden_states,
                torch.tensor(0.0, device=hidden_states.device),
            )
            final_output += self.experts[0](vision_tokens)

        # Route to text expert (1)
        text_mask = (routing_mask == 1).unsqueeze(-1)
        if text_mask.any():
            text_tokens = torch.where(
                text_mask, hidden_states, torch.tensor(0.0, device=hidden_states.device)
            )
            final_output += self.experts[1](text_tokens)

        return final_output
