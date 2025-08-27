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
from models import VisionLanguageConnector, MoELayer  # Make sure to import MoELayer
from data import COCO_Loader
from torch.cuda.amp import autocast, GradScaler


# ====================================================================================
# 0. UTIL FCN TO INSERT MOE
# ====================================================================================
def replace_ffn_with_moe(model: MistralForCausalLM) -> MistralForCausalLM:
    """Surgically replaces the FFN in each Mistral layer with our MoELayer."""
    print(" surgically replacing FFNs with MoE layers...")
    for i, layer in enumerate(model.model.layers):
        original_ffn = layer.mlp
        d_model = original_ffn.gate_proj.in_features
        moe_layer = MoELayer(model.config, d_model=d_model, num_experts=2)

        # Initialize both experts with the original FFN's weights
        moe_layer.experts[0].load_state_dict(original_ffn.state_dict())
        moe_layer.experts[1].load_state_dict(original_ffn.state_dict())

        layer.mlp = moe_layer
    print("✅ All FFN layers have been replaced with MoE layers.")
    return model


# ====================================================================================
# 1. SETUP AND CONFIGURATION
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
# 2. MODEL LOADING AND ARCHITECTURAL MODIFICATION
# ====================================================================================
print("Loading foundational models...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
tokenizer.pad_token = tokenizer.eos_token

# 1. Load the base LLM directly into 8-bit on the GPU
print("Loading and quantizing base LLM to 8-bit...")
llm = MistralForCausalLM.from_pretrained(
    paths["mistral_local_path"],
    load_in_8bit=True
)
print("✅ Base LLM loaded and quantized.")

# 2. Perform surgical replacement on the now-quantized model
llm = replace_ffn_with_moe(llm)

# Load trained Stage 1 Vision Connector
vision_connector = VisionLanguageConnector().to(DEVICE)
stage1_weights_path = os.path.join(OUTPUT_DIR, "vision_connector_stage1.pth")
if os.path.exists(stage1_weights_path):
    print(f"💾 Loading Stage 1 Vision Connector weights from {stage1_weights_path}")
    vision_connector.load_state_dict(torch.load(stage1_weights_path))
else:
    print("🚨 WARNING: Stage 1 weights not found. Connector is randomly initialized.")

# Freeze all non-expert components
print("Freezing non-expert components...")
for param in vision_encoder.parameters():
    param.requires_grad = False
for param in vision_connector.parameters():
    param.requires_grad = False
for name, param in llm.named_parameters():
    if "mlp.experts" not in name:
        param.requires_grad = False
print("✅ Components frozen.")

# ====================================================================================
# 3. DATA PIPELINE
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
# 4. TRAINING SETUP
# ====================================================================================
trainable_params = [p for p in llm.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=train_params["learning_rate"])
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
scaler = GradScaler()
loss_history = {"train": [], "val": []}
start_epoch = 0

checkpoint_dir = os.path.join(OUTPUT_DIR, "stage2_checkpoint")
if os.path.exists(checkpoint_dir):
    print(f"💾 Resuming from checkpoint: {checkpoint_dir}")
    # Reload the model state (which includes the specialized experts)
    llm = MistralForCausalLM.from_pretrained(checkpoint_dir).to(DEVICE)
    # Note: We re-apply freezing just in case, though save_pretrained should handle it
    for name, param in llm.named_parameters():
        if "mlp.experts" not in name:
            param.requires_grad = False

    # Load optimizer and scaler states
    if os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt")):
        optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        )
        scaler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scaler.pt")))

    # Load loss history and determine start epoch
    loss_path = os.path.join(OUTPUT_DIR, "loss_history_stage2.json")
    if os.path.exists(loss_path):
        with open(loss_path, "r") as f:
            loss_history = json.load(f)
        start_epoch = len(loss_history["train"])  # Resume from the next epoch

print(
    f"Optimizing {sum(p.numel() for p in trainable_params)} trainable parameters, starting from epoch {start_epoch + 1}."
)


# ====================================================================================
# 5. STAGE 2 TRAINING LOOP
# ====================================================================================
print("🚀 Starting Stage 2 training...")
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    llm.train()
    total_train_loss = 0
    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = (
            images.to(DEVICE),
            input_ids.to(DEVICE),
            attention_mask.to(DEVICE),
        )
        optimizer.zero_grad()
        with autocast():
            with torch.no_grad():
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                text_embeddings = llm.model.embed_tokens(input_ids)

            combined_embeddings = torch.cat(
                [visual_soft_tokens, text_embeddings], dim=1
            )
            routing_mask = torch.cat(
                [
                    torch.zeros(
                        visual_soft_tokens.shape[:2], dtype=torch.long, device=DEVICE
                    ),
                    torch.ones(
                        text_embeddings.shape[:2], dtype=torch.long, device=DEVICE
                    ),
                ],
                dim=1,
            )

            for layer in llm.model.layers:
                layer.mlp.routing_mask = routing_mask

            combined_attention_mask = torch.cat(
                [
                    torch.ones(visual_soft_tokens.shape[:2], device=DEVICE),
                    attention_mask,
                ],
                dim=1,
            )

            outputs = llm(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
            )
            logits = outputs.logits
            shift_logits = logits[
                ..., visual_soft_tokens.shape[1] - 1 : -1, :
            ].contiguous()
            shift_labels = input_ids.contiguous()
            loss = loss_fn(
                shift_logits.view(--1, llm.config.vocab_size), shift_labels.view(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    loss_history["train"].append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    llm.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (images, input_ids, attention_mask) in enumerate(val_loader):
            images, input_ids, attention_mask = (
                images.to(DEVICE),
                input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
            )
            with autocast():
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                text_embeddings = llm.model.embed_tokens(input_ids)

                combined_embeddings = torch.cat(
                    [visual_soft_tokens, text_embeddings], dim=1
                )
                routing_mask = torch.cat(
                    [
                        torch.zeros(
                            visual_soft_tokens.shape[:2],
                            dtype=torch.long,
                            device=DEVICE,
                        ),
                        torch.ones(
                            text_embeddings.shape[:2], dtype=torch.long, device=DEVICE
                        ),
                    ],
                    dim=1,
                )

                for layer in llm.model.layers:
                    layer.mlp.routing_mask = routing_mask

                combined_attention_mask = torch.cat(
                    [
                        torch.ones(visual_soft_tokens.shape[:2], device=DEVICE),
                        attention_mask,
                    ],
                    dim=1,
                )

                outputs = llm(
                    inputs_embeds=combined_embeddings,
                    attention_mask=combined_attention_mask,
                )
                logits = outputs.logits
                shift_logits = logits[
                    ..., visual_soft_tokens.shape[1] - 1 : -1, :
                ].contiguous()
                shift_labels = input_ids.contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, llm.config.vocab_size), shift_labels.view(-1)
                )
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    loss_history["val"].append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

    # --- Checkpointing ---
    llm.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, "scaler.pt"))
    print(f"💾 Checkpoint saved to {checkpoint_dir} after epoch {epoch+1}")

    loss_path = os.path.join(OUTPUT_DIR, "loss_history_stage2.json")
    with open(loss_path, "w") as f:
        json.dump(loss_history, f)

print("✅ Stage 2 training complete.")
