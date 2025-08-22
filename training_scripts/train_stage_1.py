import yaml
import torch
import os
import json
import gc
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
from torch.cuda.amp import autocast, GradScaler

# --- 1. Load Configuration ---
with open("./configs/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

paths = config["paths"]
train_params = config["training_stage1"]
loader_params = config["dataloader"]
NUM_EPOCHS = train_params["num_epochs"]
OUTPUT_DIR = paths["output_dir"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
if DEVICE == "cpu":
    print("WARNING: CUDA not available, using CPU!")
    exit(1)

# --- 2. Load Foundational Models ---
print("Loading foundational models...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
llm = MistralForCausalLM.from_pretrained(
    paths["mistral_local_path"], load_in_8bit=True
)
tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
tokenizer.pad_token = tokenizer.eos_token

for param in vision_encoder.parameters():
    param.requires_grad = False
for param in llm.parameters():
    param.requires_grad = False
print("✅ Models loaded and frozen.")

# --- 3. Create Datasets and DataLoaders ---
print("Creating datasets and dataloaders...")
train_dataset = COCO_Loader(
    image_dir=paths["image_dir"],
    annotations_file=paths["annotations_file"],
    clip_processor=clip_processor,
    tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"],
    split="train",
)
val_dataset = COCO_Loader(
    image_dir=paths["image_dir"],
    annotations_file=paths["annotations_file"],
    clip_processor=clip_processor,
    tokenizer=tokenizer,
    subset_fraction=train_params["subset_fraction"],
    split="val",
)
train_loader = DataLoader(
    train_dataset,
    batch_size=train_params["batch_size"],
    shuffle=True,
    num_workers=loader_params["num_workers"],
)
val_loader = DataLoader(
    val_dataset,
    batch_size=train_params["batch_size"],
    shuffle=False,
    num_workers=loader_params["num_workers"],
)

# --- 4. Setup Model, Optimizer, and Checkpoint Loading ---
vision_connector = VisionLanguageConnector().to(DEVICE)
optimizer = optim.AdamW(vision_connector.parameters(), lr=train_params["learning_rate"])
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()
loss_history = {"train": [], "val": []}

os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1.pth")

if os.path.exists(save_path):
    print(f"💾 Loading saved weights from {save_path}")
    vision_connector.load_state_dict(torch.load(save_path, weights_only=True))

# --- 5. The Training and Validation Loop ---
print("🚀 Starting training...")
for epoch in range(NUM_EPOCHS):
    # -- Training Phase --
    vision_connector.train()
    total_train_loss = 0
    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = (
            images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE)
        )
        optimizer.zero_grad()
        with autocast():
            with torch.no_grad():
                patch_embeddings = vision_encoder(images).last_hidden_state
                text_embeddings = llm.model.embed_tokens(input_ids)
            visual_soft_tokens = vision_connector(patch_embeddings)
            combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
            attention_mask = torch.cat([torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask], dim=1)
            outputs = llm(inputs_embeds=combined_embeddings, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., visual_soft_tokens.shape[1] - 1 : -1, :].contiguous()
            shift_labels = input_ids[..., :].contiguous()
            loss = loss_fn(shift_logits.view(-1, llm.config.vocab_size), shift_labels.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    loss_history["train"].append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f}")

    # -- Validation Phase --
    vision_connector.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (images, input_ids, attention_mask) in enumerate(val_loader):
            images, input_ids, attention_mask = (
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE)
            )
            with autocast():
                patch_embeddings = vision_encoder(images).last_hidden_state
                text_embeddings = llm.model.embed_tokens(input_ids)
                visual_soft_tokens = vision_connector(patch_embeddings)
                combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
                attention_mask = torch.cat([torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask], dim=1)
                outputs = llm(inputs_embeds=combined_embeddings, attention_mask=attention_mask)
                logits = outputs.logits
                shift_logits = logits[..., visual_soft_tokens.shape[1] - 1 : -1, :].contiguous()
                shift_labels = input_ids[..., :].contiguous()
                loss = loss_fn(shift_logits.view(-1, llm.config.vocab_size), shift_labels.view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    loss_history["val"].append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

    # --- Save checkpoint at the end of every epoch ---
    print(f"💾 Checkpointing model weights after epoch {epoch+1}...")
    torch.save(vision_connector.state_dict(), save_path)
    print(f"✅ Checkpoint saved to {save_path}")

    # Save loss history at the end of every epoch
    loss_path = os.path.join(OUTPUT_DIR, "loss_history_stage1.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f)

print("✅ Training complete.")