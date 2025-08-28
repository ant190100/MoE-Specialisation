import torch
import json
import os
from transformers import MistralForCausalLM, AutoConfig, AutoTokenizer

# Import your custom model class and the replacement utility
# Make sure these files are accessible (e.g., in models/ and utils/ folders)
from models.custom_mistral import MistralMoEForCausalLM
from models.utils.create_n_experts import replace_ffn_with_moe

# --- Configuration ---
base_model_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-v0.3" 
output_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE"

# 1. Load the full-precision base model
print(f"Loading base model from {base_model_path}...")
llm_base = MistralForCausalLM.from_pretrained(base_model_path)

# 2. Create an instance of your custom MoE model class
print("Creating new MistralMoEForCausalLM model instance...")
llm_moe = MistralMoEForCausalLM(llm_base.config)

# 3. Manually copy weights
print("Manually copying weights...")
# Copy all non-MLP weights (e.g., embeddings, attention layers)
llm_moe.load_state_dict(llm_base.state_dict(), strict=False)
# Manually copy the original FFN weights into BOTH experts in each layer
for layer_base, layer_moe in zip(llm_base.model.layers, llm_moe.model.layers):
    layer_moe.mlp.experts[0].load_state_dict(layer_base.mlp.state_dict())
    layer_moe.mlp.experts[1].load_state_dict(layer_base.mlp.state_dict())
print("✅ Weights successfully copied.")

# 4. Update the model's config to reflect the new architecture
llm_moe.config.model_type = "mistral_moe"
llm_moe.config.architectures = ["MistralMoEForCausalLM"]

# 5. Save the new, custom MoE model
print(f"Saving new MoE model to {output_path}...")
llm_moe.save_pretrained(output_path)

# 6. **KEY FIX**: Add auto_map to config.json for proper loading
print("Updating config.json with auto_map...")
config_path = os.path.join(output_path, "config.json")
with open(config_path, "r") as f:
    config_dict = json.load(f)

# Add auto_map to tell transformers where to find your custom classes
# Fixed format: "filename.ClassName" not "path.to.filename.ClassName"
config_dict["auto_map"] = {
    "AutoConfig": "custom_mistral.MistralMoEConfig",
    "AutoModelForCausalLM": "custom_mistral.MistralMoEForCausalLM"
}

with open(config_path, "w") as f:
    json.dump(config_dict, f, indent=2)

# 7. Copy your custom model files to the output directory
print("Copying custom model files...")
import shutil
model_files = [
    "models/custom_mistral.py",
    "models/utils/create_n_experts.py"
]
for file_path in model_files:
    if os.path.exists(file_path):
        # Copy to the root of the output directory with just the filename
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_path, filename)
        shutil.copy2(file_path, dest_path)
        print(f"Copied {file_path} to {dest_path}")

print("✅ MoE model successfully created with proper loading configuration.")