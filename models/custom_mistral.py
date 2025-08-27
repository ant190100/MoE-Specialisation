import torch.nn as nn

from transformers import MistralForCausalLM, MistralConfig
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from .moe_layer import MoELayer # Import your existing MoELayer

# 1. Define a custom configuration for your MoE model.
#    This tells transformers that any model of type "mistral_moe"
#    should use this configuration class.
class MistralMoEConfig(MistralConfig):
    model_type = "mistral_moe"

# 2. Define a custom decoder layer that uses your MoELayer.
#    This is the core of the modification. We inherit from the standard
#    MistralDecoderLayer but replace its mlp attribute with our MoELayer.
class MistralMoEDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralMoEConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace the standard FFN with your MoE layer
        self.mlp = MoELayer(config=config, d_model=config.hidden_size)
# 3. Define the final, complete model.
#    This class inherits from the standard MistralForCausalLM but
#    is hard-coded to use your custom MoEDecoderLayer for its construction.
class MistralMoEForCausalLM(MistralForCausalLM):
    config_class = MistralMoEConfig

    def __init__(self, config):
        super().__init__(config)
        # Overwrite the standard layers with your custom MoE layers
        self.model.layers = nn.ModuleList(
            [MistralMoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
