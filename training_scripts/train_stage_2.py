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
    AutoConfig,
    AutoModelForCausalLM
)
from models import VisionLanguageConnector
from data import COCO_Loader
from torch.cuda.amp import autocast, GradScaler

# Import your custom MoE classes
from models.custom_mistral import MistralMoEConfig, MistralMoEForCausalLM

# --- 1. Register Your Custom Architecture ---
print("Registering custom MistralMoE architecture...")
AutoConfig.register("mistral_moe", MistralMoEConfig)
AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)
print("✅ Custom architecture registered.")

# ====================================================================================
# 2. SETUP AND CONFIGURATION
# ====================================================================================
print("--- Initializing Stage 2 Training ---")

with open("./configs/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

paths = config["paths"]
train_params = config["training_stage2"]
loader_params = config["dataloader"]
NUM_EPOCHS = train_params["num_epochs"]
OUTPUT_DIR = paths["output_dir"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ====================================================================================
# 3. MODEL LOADING
# ====================================================================================
print("Loading foundational models...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
tokenizer.pad_token = tokenizer.eos_token

# --- Simplified and Robust Loading ---
# This path should point to the MoE model you created with the 'create_moe_model.py' script
moe_model_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE"

print(f"Loading custom MoE model from {moe_model_path}...")
llm = AutoModelForCausalLM.from_pretrained(
    moe_model_path,
    trust_remote_code=True, # Trust your custom model files
    local_files_only=True
).to(DEVICE)
print("✅ Custom MoE model loaded and quantized.")

# --- Load Trained Stage 1 Vision Connector ---
vision_connector = VisionLanguageConnector().to(DEVICE)
stage1_weights_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1.pth")
if os.path.exists(stage1_weights_path):
    print(f"💾 Loading Stage 1 Vision Connector weights from {stage1_weights_path}")
    vision_connector.load_state_dict(torch.load(stage1_weights_path))
else:
    print("🚨 WARNING: Stage 1 weights not found.")

# ====================================================================================
# 4. DATA PIPELINE
# ====================================================================================
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

# ====================================================================================
# 5. TRAINING SETUP
# ====================================================================================
# --- Freeze/Unfreeze Parameters ---
print("Preparing model for Stage 2 training...")
for param in vision_encoder.parameters(): param.requires_grad = False
for param in vision_connector.parameters(): param.requires_grad = False
for param in llm.parameters(): param.requires_grad = False
for layer in llm.model.layers:
    for expert in layer.mlp.experts:
        for param in expert.parameters():
            param.requires_grad = True
print("✅ All components frozen except MoE experts.")

# --- Optimizer and Loss Setup ---
trainable_params = [p for p in llm.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=train_params["learning_rate"])
scaler = GradScaler()
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

print(f"Optimizing {sum(p.numel() for p in trainable_params)} trainable parameters.")

# ====================================================================================
# 6. STAGE 2 TRAINING LOOP
# ====================================================================================
print(f"🚀 Starting Stage 2 training for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    llm.train()
    total_train_loss = 0
    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = (
            images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE)
        )
        optimizer.zero_grad()
        with autocast():
            with torch.no_grad():
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                text_embeddings = llm.model.embed_tokens(input_ids)
            combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
            routing_mask = torch.cat([
                torch.zeros(visual_soft_tokens.shape[:2], dtype=torch.long, device=DEVICE),
                torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=DEVICE)
            ], dim=1)
            for layer in llm.model.layers:
                layer.mlp.routing_mask = routing_mask
            combined_attention_mask = torch.cat([
                torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask
            ], dim=1)
            outputs = llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., visual_soft_tokens.shape[1] - 1 : -1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, llm.config.vocab_size), shift_labels.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    llm.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (images, input_ids, attention_mask) in enumerate(val_loader):
            images, input_ids, attention_mask = (
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE)
            )
            with autocast():
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                text_embeddings = llm.model.embed_tokens(input_ids)
                combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
                routing_mask = torch.cat([
                    torch.zeros(visual_soft_tokens.shape[:2], dtype=torch.long, device=DEVICE),
                    torch.ones(text_embeddings.shape[:2], dtype=torch.long, device=DEVICE)
                ], dim=1)
                for layer in llm.model.layers:
                    layer.mlp.routing_mask = routing_mask
                combined_attention_mask = torch.cat([
                    torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask
                ], dim=1)
                outputs = llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
                logits = outputs.logits
                shift_logits = logits[..., visual_soft_tokens.shape[1] - 1 : -1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, llm.config.vocab_size), shift_labels.view(-1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

print("✅ Training complete.")
