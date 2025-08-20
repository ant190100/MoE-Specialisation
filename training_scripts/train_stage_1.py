import yaml
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer, MistralForCausalLM, CLIPVisionModel

from models import VisionLanguageConnector
from data import COCO_Loader

# --- 1. Load Configuration from YAML file ---
with open('./configs/training_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract config values into variables for easy access
paths = config['paths']
train_params = config['training_stage1']
loader_params = config['dataloader']
NUM_EPOCHS = train_params['num_epochs']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Load Models, Processors, and Tokenizer from Local Paths ---
print("Loading foundational models from local paths...")
vision_encoder = CLIPVisionModel.from_pretrained(paths['clip_local_path']).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(paths['clip_local_path'])
llm = MistralForCausalLM.from_pretrained(paths['mistral_local_path']).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(paths['mistral_local_path'])
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
dataset = COCO_Loader(
    image_dir=paths['image_dir'],
    annotations_file=paths['annotations_file'],
    clip_processor=clip_processor,
    tokenizer=tokenizer,
    subset_fraction=train_params['subset_fraction']
)
train_loader = DataLoader(
    dataset,
    batch_size=train_params['batch_size'],
    shuffle=True,
    num_workers=loader_params['num_workers'],
    pin_memory=True
)

# --- 4. Setup Optimizer and Loss ---
optimizer = optim.AdamW(vision_connector.parameters(), lr=train_params['learning_rate'])
loss_fn = nn.CrossEntropyLoss()

# --- 5. The Training Loop ---
print("🚀 Starting Stage 1 training...")
for epoch in range(NUM_EPOCHS):
    vision_connector.train()
    total_loss = 0
    
    for i, (images, input_ids, attention_mask) in enumerate(train_loader):
        images, input_ids, attention_mask = images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE)
        optimizer.zero_grad()

        with torch.no_grad():
            visual_outputs = vision_encoder(pixel_values=images, output_hidden_states=False)
            patch_embeddings = visual_outputs.last_hidden_state
            text_embeddings = llm.model.embed_tokens(input_ids)

        visual_soft_tokens = vision_connector(patch_embeddings)
        combined_embeddings = torch.cat([visual_soft_tokens, text_embeddings], dim=1)
        
        visual_attention_mask = torch.ones(visual_soft_tokens.shape[:2], dtype=torch.long, device=DEVICE)
        combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
        
        outputs = llm(inputs_embeds=combined_embeddings, attention_mask=combined_attention_mask)
        logits = outputs.logits
        
        # Calculate loss only on the caption tokens
        shift_logits = logits[..., visual_soft_tokens.shape[1]:-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = loss_fn(shift_logits.view(-1, llm.config.vocab_size), shift_labels.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

print("✅ Stage 1 training complete.")

# --- Save the Trained Connector Weights ---
print("💾 Saving the trained vision connector weights...")

# Create the main output directory if it doesn't exist
output_dir = paths['output_dir']
os.makedirs(output_dir, exist_ok=True)

# Define a specific path for the Stage 1 weights
save_path = os.path.join(output_dir, 'vision_connector_stage1.pth')

# Save the model's state dictionary
torch.save(vision_connector.state_dict(), save_path)

print(f"✅ Connector weights saved to {save_path}")
