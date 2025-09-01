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
        
        # --- This is the standard, correct implementation ---

        # 1. Get the output from each expert for all tokens.
        #    This is required for gradient checkpointing to work correctly.
        vision_output = self.experts[0](hidden_states)
        text_output = self.experts[1](hidden_states)

        # 2. Create the routing masks (0s and 1s) and ensure they have the same dtype as the outputs.
        vision_mask = (routing_mask == 0).unsqueeze(-1).to(vision_output.dtype)
        text_mask = (routing_mask == 1).unsqueeze(-1).to(text_output.dtype)

        # 3. Multiply each expert's output by its mask.
        #    This zeros out the tokens that don't belong to that expert.
        #    The gradient path is maintained.
        vision_contribution = vision_output * vision_mask
        text_contribution = text_output * text_mask

        # 4. Sum the contributions. Since masks are mutually exclusive, this is equivalent
        #    to selecting the correct expert output for each token.
        final_output = vision_contribution + text_contribution

        return final_output
