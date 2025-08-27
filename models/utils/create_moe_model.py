import torch
from transformers import MistralForCausalLM, AutoConfig
import os

# Import your custom model class and the replacement utility
# Make sure these files are accessible (e.g., in models/ and utils/ folders)
from models.custom_mistral import MistralMoEForCausalLM
from utils import replace_ffn_with_moe

# --- Configuration ---
# Point to your downloaded base Mistral 7B model
base_model_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-v0.3" 
# Define where the new, custom MoE model will be saved
output_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE"

# 1. Load the full-precision base model
print(f"Loading base model from {base_model_path}...")
llm_base = MistralForCausalLM.from_pretrained(base_model_path)

# 2. Perform the surgical replacement
llm_moe = replace_ffn_with_moe(llm_base)

# 3. Create an instance of your custom MoE model class
print("Creating new MistralMoEForCausalLM model instance...")
# Initialize with the config of the now-modified base model
final_moe_model = MistralMoEForCausalLM(llm_moe.config)

# 4. Load the modified weights into your custom model instance
final_moe_model.load_state_dict(llm_moe.state_dict())
print("✅ Weights loaded into custom architecture.")

# 5. CRITICAL: Update the model's config to reflect the new architecture
final_moe_model.config.model_type = "mistral_moe"
final_moe_model.config.architectures = ["MistralMoEForCausalLM"]

# 6. Save the new, custom MoE model
print(f"Saving new MoE model to {output_path}...")
final_moe_model.save_pretrained(output_path)
print("✅ MoE model successfully created.")