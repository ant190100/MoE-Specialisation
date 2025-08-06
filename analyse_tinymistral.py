# ==============================================================================
#
# Analysis Script for TinyMistral-6x248M
#
# Description:
# This script loads the pre-trained TinyMistral MoE model and runs a
# diagnostic analysis on it using the 20 Newsgroups dataset.
# It does not perform any training.
#
# ==============================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os

# --- 1. Configuration ---
# IMPORTANT: Point this to the local path where you downloaded the model
MODEL_PATH = os.path.join(os.path.expanduser('~'), "models", "TinyMistral-6x248M")
NUM_EXPERTS = 6
NUM_CLASSES = 20
BATCH_SIZE = 16 # Adjust based on GPU memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- 2. Load Model and Data ---
print(f"Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, # Use bfloat16 for better performance on modern GPUs
    device_map="auto"
)
model.eval() # Set model to evaluation mode

# Load dataset
print("Preparing 20 Newsgroups data...")
dataset = load_dataset("SetFit/20_newsgroups", split="test")
class_names = [
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
    'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
    'sci.space', 'soc.religion.christian', 'talk.politics.guns',
    'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
]

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs['labels'] = torch.tensor(labels)
    return inputs

test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# --- 3. Ablation Analysis Function ---
def run_ablation_analysis(model, test_loader, device, num_classes, num_experts, class_names):
    print("\n--- Starting Ablation Analysis ---")
    
    def evaluate_per_class_accuracy():
        class_correct = [0] * num_classes
        class_totals = [0] * num_classes
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # This model is a Causal LM, so we need a different way to get a class prediction.
                # A simple approach is to use the model's output logits for the last token.
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :] # Logits of the last token
                
                # We need a classification head. For simplicity, we'll project to num_classes.
                # In a real scenario, you might use a more sophisticated method.
                classifier_head = torch.nn.Linear(logits.size(-1), num_classes).to(device)
                class_logits = classifier_head(logits)
                
                _, predicted = torch.max(class_logits.data, 1)
                
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_totals[label] += 1
        
        accuracies = [(class_correct[i] / class_totals[i]) * 100 if class_totals[i] > 0 else 0 for i in range(num_classes)]
        return accuracies

    print("Calculating baseline accuracy...")
    baseline_accuracies = evaluate_per_class_accuracy()
    print("\nBaseline Per-Class Accuracy (%):")
    baseline_df = pd.DataFrame([baseline_accuracies], columns=[name[:15] for name in class_names], index=['Baseline Acc.'])
    print(baseline_df.round(2))

    ablation_results = []
    # This model has MoE layers at every other layer, starting from the first.
    # We will just ablate the experts in the first MoE layer as a proof of concept.
    moe_layer_index = 0 
    
    for expert_to_ablate in range(num_experts):
        print(f"\nAblating Expert {expert_to_ablate} in MoE Layer {moe_layer_index}...")
        
        expert_path = model.model.layers[moe_layer_index].block_sparse_moe.experts[expert_to_ablate]
        original_weights = [p.clone().detach() for p in expert_path.parameters()]

        with torch.no_grad():
            for param in expert_path.parameters():
                param.data.fill_(0)
        
        ablated_accuracies = evaluate_per_class_accuracy()
        ablation_results.append(ablated_accuracies)

        with torch.no_grad():
            for i, param in enumerate(expert_path.parameters()):
                param.data.copy_(original_weights[i])
        print(f"Restored Expert {expert_to_ablate}.")

    print("\n--- Ablation Analysis Results ---")
    accuracy_drops = [[base - ablated for base, ablated in zip(baseline_accuracies, expert_accuracies)] for expert_accuracies in ablation_results]
    df = pd.DataFrame(accuracy_drops, columns=[name[:15] for name in class_names], index=[f"Ablate Expert {i}" for i in range(num_experts)])
    print("\nAccuracy Drop (%) After Ablating Each Expert:")
    pd.set_option('display.width', 1000)
    print(df.round(2))

# --- 4. Main Execution ---
if __name__ == "__main__":
    run_ablation_analysis(model, test_loader, DEVICE, NUM_CLASSES, NUM_EXPERTS, class_names)
