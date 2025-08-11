#!/usr/bin/env python3
"""
Debug TinyMistral routing - investigate the actual MoE structure
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def debug_tinymistral_routing():
    print("🔍 DEBUGGING TINYMISTRAL ROUTING STRUCTURE")
    print("=" * 50)
    
    # Load model
    print("Loading TinyMistral...")
    model = AutoModelForCausalLM.from_pretrained("M4-ai/TinyMistral-6x248M", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("M4-ai/TinyMistral-6x248M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Examine the structure
    print(f"\nModel structure:")
    print(f"Number of layers: {len(model.model.layers)}")
    
    # Look at first layer in detail
    first_layer = model.model.layers[0]
    print(f"\nFirst layer attributes: {dir(first_layer)}")
    
    if hasattr(first_layer, 'block_sparse_moe'):
        moe = first_layer.block_sparse_moe
        print(f"\nMoE attributes: {dir(moe)}")
        print(f"MoE type: {type(moe)}")
        
        if hasattr(moe, 'gate'):
            print(f"Gate type: {type(moe.gate)}")
            print(f"Gate weight shape: {moe.gate.weight.shape}")
        
        if hasattr(moe, 'experts'):
            print(f"Experts type: {type(moe.experts)}")
            print(f"Number of experts: {len(moe.experts)}")
    
    # Test with actual data
    print(f"\n🧪 TESTING WITH REAL DATA")
    dataset = load_dataset("ag_news", split="test")
    test_texts = dataset["text"][:2]
    
    # Tokenize
    inputs = tokenizer(test_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Hook to capture routing
    routing_data = {}
    
    def capture_routing(name):
        def hook(module, input, output):
            print(f"\n--- Hook called for {name} ---")
            print(f"Module type: {type(module)}")
            print(f"Input type: {type(input)}")
            print(f"Input length: {len(input) if isinstance(input, (list, tuple)) else 'not sequence'}")
            
            if len(input) > 0:
                inp = input[0]
                print(f"First input shape: {inp.shape if hasattr(inp, 'shape') else type(inp)}")
                
                # Try to get routing logits
                if hasattr(module, 'gate') and hasattr(inp, 'shape'):
                    try:
                        # Reshape input
                        batch_size, seq_len, hidden_dim = inp.shape
                        inp_flat = inp.view(-1, hidden_dim)
                        print(f"Flattened input shape: {inp_flat.shape}")
                        
                        # Get routing logits
                        routing_logits = module.gate(inp_flat)
                        print(f"Routing logits shape: {routing_logits.shape}")
                        
                        # Get probabilities
                        routing_probs = torch.softmax(routing_logits, dim=-1)
                        print(f"Routing probs shape: {routing_probs.shape}")
                        
                        # Show some statistics
                        expert_counts = torch.argmax(routing_probs, dim=-1)
                        unique, counts = torch.unique(expert_counts, return_counts=True)
                        print(f"Expert usage: {dict(zip(unique.tolist(), counts.tolist()))}")
                        
                        # Store data
                        routing_data[name] = {
                            'logits': routing_logits.detach(),
                            'probs': routing_probs.detach(),
                            'expert_counts': expert_counts.detach()
                        }
                        
                    except Exception as e:
                        print(f"Error in routing capture: {e}")
            
            print(f"Output type: {type(output)}")
            if hasattr(output, 'shape'):
                print(f"Output shape: {output.shape}")
            elif isinstance(output, (list, tuple)):
                print(f"Output length: {len(output)}")
                if len(output) > 0 and hasattr(output[0], 'shape'):
                    print(f"First output shape: {output[0].shape}")
        
        return hook
    
    # Register hooks on first few layers
    hooks = []
    for i in range(min(3, len(model.model.layers))):
        layer = model.model.layers[i]
        if hasattr(layer, 'block_sparse_moe'):
            hook = layer.block_sparse_moe.register_forward_hook(capture_routing(f"layer_{i}"))
            hooks.append(hook)
    
    print(f"\n🚀 Running forward pass...")
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            print(f"Forward pass successful!")
            print(f"Output logits shape: {outputs.logits.shape}")
        except Exception as e:
            print(f"Forward pass error: {e}")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze routing data
    print(f"\n📊 ROUTING ANALYSIS")
    for layer_name, data in routing_data.items():
        probs = data['probs']
        expert_counts = data['expert_counts']
        
        print(f"\n{layer_name}:")
        print(f"  Total tokens: {len(expert_counts)}")
        
        # Expert utilization
        unique, counts = torch.unique(expert_counts, return_counts=True)
        total_tokens = len(expert_counts)
        
        print(f"  Expert utilization:")
        for expert_id, count in zip(unique.tolist(), counts.tolist()):
            percentage = (count / total_tokens) * 100
            print(f"    Expert {expert_id}: {count} tokens ({percentage:.1f}%)")
        
        # Show some routing probabilities
        print(f"  Sample routing probabilities (first 5 tokens):")
        for i in range(min(5, len(probs))):
            probs_str = ", ".join([f"{p:.3f}" for p in probs[i].tolist()])
            print(f"    Token {i}: [{probs_str}]")

if __name__ == "__main__":
    debug_tinymistral_routing()
