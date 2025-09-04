import time
import json
import yaml
import torch
import os
import gc
import sys
import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    CLIPVisionModel,
)
from models import VisionLanguageConnector
from data import COCO_Loader
from torch.amp import GradScaler, autocast
from torch.distributed.fsdp import CPUOffload
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from models.custom_mistral import (
    MistralMoEConfig,
    MistralMoEForCausalLM,
    MistralMoEDecoderLayer,
)
from transformers.models.mistral.modeling_mistral import MistralMLP

# --- 1. Register Custom Architecture ---
AutoConfig.register("mistral_moe", MistralMoEConfig)
AutoModelForCausalLM.register(MistralMoEConfig, MistralMoEForCausalLM)

# ====================================================================================
# 2. SETUP AND CONFIGURATION
# ====================================================================================
with open("./configs/training_config.yaml", "r") as file:
    config = yaml.safe_load(file)

paths = config["paths"]
train_params = config["training_stage2.5"]
loader_params = config["dataloader"]
NUM_EPOCHS = train_params["num_epochs"]
OUTPUT_DIR = paths["output_dir"]
STAGE2_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "stage2_checkpoints")
STAGE2_5_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "stage2_5_checkpoints")
LOAD_BALANCING_COEFF = train_params.get("load_balancing_coeff", 0.01)

# --- Initialize the distributed environment ---
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
DEVICE = local_rank

if local_rank == 0:
    print("--- Initializing Stage 2.5 Training (Training the Router) ---")

# ====================================================================================
# 3. MODEL LOADING
# ====================================================================================
if local_rank == 0:
    print("Loading foundational models...")
vision_encoder = CLIPVisionModel.from_pretrained(paths["clip_local_path"]).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths["clip_local_path"])
tokenizer = AutoTokenizer.from_pretrained(paths["mistral_local_path"])
tokenizer.pad_token = tokenizer.eos_token

moe_model_path = "/data/gpfs/projects/COMP90055/aticinovic/models/Mistral-7B-MoE"
llm = AutoModelForCausalLM.from_pretrained(
    moe_model_path,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# ====================================================================================
# 4. TRAINING SETUP
# ====================================================================================

# --- 4.1. Set Routing Mode ---
# Explicitly set all MoE layers to 'soft' routing mode.
if local_rank == 0:
    print("Setting MoE layers to 'soft' routing mode for Stage 2.5.")
for layer in llm.model.layers:
    if hasattr(layer.mlp, "routing_mode"):
        layer.mlp.routing_mode = 'soft'

# --- 4.2. Parameter Freezing ---
# **DIFFERENCE**: Freeze everything except the router 'gate' parameters.
if local_rank == 0:
    print("Preparing model for Stage 2.5: Freezing experts, unfreezing routers.")
for name, param in llm.named_parameters():
    if "mlp.gate" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# ====================================================================================
# 5. FSDP WRAPPING & CHECKPOINTING
# ====================================================================================
my_auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={MistralMLP})
ignored_modules = [llm.model.embed_tokens]

llm = FSDP(
    llm,
    device_id=DEVICE,
    auto_wrap_policy=my_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
    ),
    use_orig_params=True,
    ignored_modules=ignored_modules,
)

# --- 5.1. Load Stage 2 Checkpoint (Base weights for experts) ---
# Find the latest completed epoch from Stage 2 to load as a base.
latest_stage2_epoch = 0
if local_rank == 0:
    if os.path.exists(STAGE2_CHECKPOINT_DIR):
        epoch_numbers = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in os.listdir(STAGE2_CHECKPOINT_DIR) if re.search(r'epoch_(\d+)', f)]
        if epoch_numbers:
            latest_stage2_epoch = max(epoch_numbers)

epoch_tensor = torch.tensor([latest_stage2_epoch], dtype=torch.int).to(DEVICE)
dist.broadcast(epoch_tensor, src=0)
latest_stage2_epoch = epoch_tensor.item()

if latest_stage2_epoch > 0:
    checkpoint_path = os.path.join(STAGE2_CHECKPOINT_DIR, f"llm_stage2_epoch_{latest_stage2_epoch}.pth")
    if local_rank == 0:
        print(f"💾 Loading Stage 2 expert weights from: {checkpoint_path}")
    
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        llm.load_state_dict(state_dict, strict=False) # Use strict=False as optimizer state won't match
    del state_dict
    gc.collect()
    if local_rank == 0:
        print("✅ Stage 2 expert weights loaded successfully.")
else:
    if local_rank == 0:
        print("🚨 WARNING: No Stage 2 checkpoint found. Routers will be trained on unspecialized experts.")

# --- 5.2. Resume from Stage 2.5 Checkpoint (If it exists) ---
latest_epoch = 0
if local_rank == 0:
    if os.path.exists(STAGE2_5_CHECKPOINT_DIR):
        epoch_numbers = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in os.listdir(STAGE2_5_CHECKPOINT_DIR) if re.search(r'epoch_(\d+)', f)]
        if epoch_numbers:
            latest_epoch = max(epoch_numbers)

epoch_tensor = torch.tensor([latest_epoch], dtype=torch.int).to(DEVICE)
dist.broadcast(epoch_tensor, src=0)
latest_epoch = epoch_tensor.item()

# This loading logic is already correct and expects a full checkpoint.
if latest_epoch > 0:
    checkpoint_path = os.path.join(STAGE2_5_CHECKPOINT_DIR, f"llm_stage2_5_epoch_{latest_epoch}.pth")
    if local_rank == 0:
        print(f"💾 Resuming Stage 2.5 training from epoch {latest_epoch+1} using {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, load_policy):
        llm.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    del checkpoint
    gc.collect()
    if local_rank == 0:
        print("✅ Stage 2.5 training state resumed successfully.")

# --- 5.3. Finalize Model Setup ---
llm.model.embed_tokens.to(DEVICE)
llm.gradient_checkpointing_enable()
vision_connector = VisionLanguageConnector().to(DEVICE)
vision_connector.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "vision_connector_stage1.pth"), map_location=DEVICE))
for param in vision_encoder.parameters(): param.requires_grad = False
for param in vision_connector.parameters(): param.requires_grad = False

# ====================================================================================
# 6. DATA & OPTIMIZER
# ====================================================================================
if local_rank == 0:
    print("Creating datasets and dataloaders...")
train_dataset = COCO_Loader(image_dir=paths["image_dir"], annotations_file=paths["annotations_file"], clip_processor=clip_processor, tokenizer=tokenizer, subset_fraction=train_params["subset_fraction"], split="train")
val_dataset = COCO_Loader(image_dir=paths["image_dir"], annotations_file=paths["annotations_file"], clip_processor=clip_processor, tokenizer=tokenizer, subset_fraction=train_params["subset_fraction"], split="val")
train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], sampler=train_sampler, num_workers=loader_params["num_workers"], pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=train_params["batch_size"], sampler=val_sampler, num_workers=loader_params["num_workers"], pin_memory=True)

accumulation_steps = train_params.get("gradient_accumulation_steps", 1)
trainable_params = [p for p in llm.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=train_params["learning_rate"], weight_decay=train_params["weight_decay"], fused=True)
scaler = GradScaler()
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

if local_rank == 0:
    print(f"Optimizing {sum(p.numel() for p in trainable_params)} trainable router parameters.")

metrics_history = {"epoch": [], "train_loss": [], "val_loss": [], "learning_rate":[]}
metrics_path = os.path.join(OUTPUT_DIR, "training_metrics_stage2.5.json")
if local_rank == 0 and latest_epoch > 0 and os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_history = json.load(f)

# ====================================================================================
# 8. TRAINING LOOP
# ====================================================================================
if local_rank == 0:
    print(f"🚀 Starting Stage 2.5 training from epoch {latest_epoch+1}...")
for epoch in range(latest_epoch, NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    llm.train()
    total_train_loss = 0
    optimizer.zero_grad()
    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = (images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE))

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                text_embeddings = llm.model.embed_tokens(input_ids)
            
            combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
            combined_attention_mask = torch.cat([torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask], dim=1)

            # **DIFFERENCE**: No manual routing mask needed. The model handles it internally.
            outputs = llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
            logits = outputs.logits

            # Calculate Cross-Entropy loss
            num_visual_tokens = visual_soft_tokens.shape[1]
            text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
            text_labels = input_ids[..., 1:].contiguous()
            ce_loss = loss_fn(text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1))

            # **DIFFERENCE**: Collect and add the load balancing loss.
            total_load_balancing_loss = 0
            for layer in llm.module.model.layers:
                if hasattr(layer.mlp, "load_balancing_loss"):
                    total_load_balancing_loss += layer.mlp.load_balancing_loss
            
            loss = (ce_loss + LOAD_BALANCING_COEFF * total_load_balancing_loss) / accumulation_steps

        scaler.scale(loss).backward()
        if loss.item() > 0:
            total_train_loss += loss.item() * accumulation_steps

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if local_rank == 0 and (i + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}, Batch [{i+1}/{len(train_loader)}]")

    avg_train_loss = total_train_loss / len(train_loader)
    if local_rank == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    llm.eval()
    total_val_loss = 0
    with torch.no_grad():
        for i, (images, input_ids, attention_mask) in enumerate(val_loader):
            images, input_ids, attention_mask = (images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE))
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                patch_embeddings = vision_encoder(images).last_hidden_state
                visual_soft_tokens = vision_connector(patch_embeddings)
                text_embeddings = llm.model.embed_tokens(input_ids)
                combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
                combined_attention_mask = torch.cat([torch.ones(visual_soft_tokens.shape[:2], device=DEVICE), attention_mask], dim=1)
                
                outputs = llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
                logits = outputs.logits

                num_visual_tokens = visual_soft_tokens.shape[1]
                text_logits = logits[..., num_visual_tokens:-1, :].contiguous()
                text_labels = input_ids[..., 1:].contiguous()
                loss = loss_fn(text_logits.view(-1, llm.config.vocab_size), text_labels.view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    if local_rank == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Loss: {avg_val_loss:.4f}")

    # --- Metrics and Checkpoint Saving ---
    if local_rank == 0:
        metrics_history["epoch"].append(epoch + 1)
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        with open(metrics_path, "w") as f:
            json.dump(metrics_history, f, indent=4)
        print(f"✅ Metrics saved to {metrics_path}")

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(llm, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = llm.state_dict()

    if local_rank == 0:
        # FIX: Create a comprehensive checkpoint dictionary that includes all necessary states.
        checkpoint = {
            'model_state_dict': cpu_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch + 1
        }
        
        os.makedirs(STAGE2_5_CHECKPOINT_DIR, exist_ok=True)
        file_path = os.path.join(STAGE2_5_CHECKPOINT_DIR, f"llm_stage2_5_epoch_{epoch+1}.pth")
        
        # FIX: Save the entire checkpoint dictionary, not just the model weights.
        torch.save(checkpoint, file_path)
        print(f"Router checkpoint saved to {file_path}")
        
        previous_checkpoint_path = os.path.join(STAGE2_5_CHECKPOINT_DIR, f"llm_stage2_5_epoch_{epoch}.pth")
        if os.path.exists(previous_checkpoint_path):
            os.remove(previous_checkpoint_path)
            print(f"Removed previous checkpoint: {previous_checkpoint_path}")
    dist.barrier()

dist.destroy_process_group()
print("Job finished.")