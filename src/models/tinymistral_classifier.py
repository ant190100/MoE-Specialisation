"""
TinyMistral MoE model adapted for classification tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any

class TinyMistralForClassification(nn.Module):
    """
    Wrapper for TinyMistral-6x248M MoE model to work with classification tasks.
    """
    
    def __init__(self, model_name: str = "M4-ai/TinyMistral-6x248M", num_classes: int = 4):
        super().__init__()
        
        print(f"Loading TinyMistral model: {model_name}")
        print("⏳ This may take a while for the first download...")
        
        # Load the base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Get model configuration
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        
        # Add classification head
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Store MoE information
        self.num_experts = getattr(self.config, 'num_local_experts', 6)  # TinyMistral has 6 experts
        self.num_classes = num_classes
        
        # For tracking routing weights during forward pass
        self.last_routing_weights = None
        self.expert_outputs = None
        
        print(f"✅ TinyMistral loaded: {self.num_experts} experts, {num_classes} classes")
        print(f"📊 Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass adapted for classification.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False  # Disable cache for classification
        )
        
        # Extract last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Use the last token for classification (similar to GPT approach)
        # Find the last non-padded token for each sequence
        if attention_mask is not None:
            batch_size = input_ids.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            pooled_hidden = last_hidden_state[range(batch_size), sequence_lengths]
        else:
            # If no attention mask, use the last token
            pooled_hidden = last_hidden_state[:, -1, :]
        
        # Store routing information for analysis (if available)
        if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
            self.last_routing_weights = outputs.router_logits
            
        # Classification
        logits = self.classifier(pooled_hidden)
        
        return logits
    
    def get_routing_weights(self) -> Optional[torch.Tensor]:
        """Get the last routing weights for analysis."""
        return self.last_routing_weights
    
    def get_expert_info(self) -> Dict[str, Any]:
        """Get information about the MoE structure."""
        return {
            'num_experts': self.num_experts,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'model_name': 'TinyMistral-6x248M'
        }

def load_tinymistral_tokenizer(model_name: str = "M4-ai/TinyMistral-6x248M") -> AutoTokenizer:
    """Load the TinyMistral tokenizer with proper configuration."""
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left'  # For causal LM
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer
