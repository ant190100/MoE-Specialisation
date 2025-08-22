import yaml
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    MistralForCausalLM,
    CLIPVisionModel,
)

from models import VisionLanguageConnector
from data import COCO_Loader
# Import both autocast and GradScaler
from torch.cuda.amp import autocast, GradScaler

# --- 1. Load Configuration from YAML file ---
with open("./configs/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract config values into variables for easy access
paths = config["paths"]
train_params = config["training_stage1"]
loader_params = config["dataloader"]
NUM_EPOCHS = train_params["num_epochs"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if DEVICE == "cpu":
    print("WARNING: CUDA not available, using CPU - this will use lots of RAM!")
    exit(1)  # Stop execution if CUDA isn't working

# Verify GPU is available
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

# --- 2. Load Models, Processors, and Tokenizer from Local Paths ---
print("Loading foundational models from local paths...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
llm = MistralForCausalLM.from_pretrained(
    paths["mistral_local_path"],
    load_in_8bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
tokenizer.pad_token = tokenizer.eos_token

# Freeze model weights
for param in vision_encoder.parameters():
    param.requires_grad = False
for param in llm.parameters():
    param.requires_grad = False
print("✅ Models loaded and frozen.")

# --- 3. Instantiate Your Custom Components ---
print("Instantiating custom components...")
vision_connector = VisionLanguageConnector().to(DEVICE)

weights_path = "/data/gpfs/projects/COMP90055/aticinovic/outputs/vision_connector_stage1.pth"
if os.path.exists(weights_path):
    print(f"Loading saved weights from {weights_path}")
    vision_connector.load_state_dict(torch.load(weights_path))
    
dataset = COCO_Loader(
    image_dir=paths["image_dir"],
    annotations_file=paths["annotations_file"],
    clip_processor=clip_processor,
    tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"],
)
train_loader = DataLoader(
    dataset,
    batch_size=train_params["batch_size"],
    shuffle=True,
    num_workers=loader_params["num_workers"],
    pin_memory=True,
)

# --- 4. Setup Optimizer, Loss, and GradScaler ---
optimizer = optim.AdamW(vision_connector.parameters(), lr=train_params["learning_rate"])
loss_fn = nn.CrossEntropyLoss()
# Initialize the gradient scaler for mixed-precision training
scaler = GradScaler()

# --- 5. The Training Loop ---
print("🚀 Starting Stage 1 training...")
for epoch in range(NUM_EPOCHS):
    vision_connector.train()
    total_loss = 0

    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = (
            images.to(DEVICE),
            input_ids.to(DEVICE),
            attention_mask.to(DEVICE),
        )
        optimizer.zero_grad()

        # Wrap the forward pass with autocast
        with autocast():
            with torch.no_grad():
                visual_outputs = vision_encoder(
                    pixel_values=images, output_hidden_states=False
                )
                patch_embeddings = visual_outputs.last_hidden_state
                text_embeddings = llm.model.embed_tokens(input_ids)

            visual_soft_tokens = vision_connector(patch_embeddings)
            combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)

            visual_attention_mask = torch.ones(
                visual_soft_tokens.shape[:2], dtype=torch.long, device=DEVICE
            )
            combined_attention_mask = torch.cat(
                [visual_attention_mask, attention_mask], dim=1
            )

            outputs = llm(
                inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask
            )
            logits = outputs.logits

            # Calculate loss only on the caption tokens
            shift_logits = logits[..., visual_soft_tokens.shape[1] : -1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, llm.config.vocab_size), shift_labels.view(-1)
            )
        
        # Scale the loss, perform backward pass, and update weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

        # Clean up intermediate tensors
        del visual_outputs, patch_embeddings, text_embeddings, visual_soft_tokens
        del combined_embeddings, visual_attention_mask, combined_attention_mask
        del outputs, logits, shift_logits, shift_labels, loss

    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")
    # Add memory cleanup between epochs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

print("✅ Stage 1 training complete.")

# --- Free up memory before saving ---
print("🧹 Cleaning up memory...")
del llm, vision_encoder  # Remove large models from memory
torch.cuda.empty_cache() if torch.cuda.is_available() else None
import gc

gc.collect()

# --- Save the Trained Connector Weights ---
print("💾 Saving the trained vision connector weights...")

# Create the main output directory if it doesn't exist
output_dir = paths["output_dir"]
os.makedirs(output_dir, exist_ok=True)

# Define a specific path for the Stage 1 weights
save_path = os.path.join(output_dir, "vision_connector_stage1.pth")

# Save the model's state dictionary
torch.save(vision_connector.state_dict(), save_path)

print(f"✅ Connector weights saved to {save_path}")