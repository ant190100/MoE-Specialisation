"""
TinyMistral Classification Model
A wrapper around the pre-trained TinyMistral model for classification tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class TinyMistralForClassification(nn.Module):
    """TinyMistral model adapted for classification tasks."""

    def __init__(self, model_name="M4-ai/TinyMistral-6x248M", num_classes=4):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Load pre-trained TinyMistral model
        logger.info(f"Loading TinyMistral model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )

        # Add classification head
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        logger.info(f"✅ TinyMistral classifier created with {num_classes} classes")

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model."""
        # Get hidden states from the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last layer hidden states
        last_hidden_states = outputs.hidden_states[
            -1
        ]  # [batch_size, seq_len, hidden_size]

        # Pool the sequence (use last token or mean pooling)
        if attention_mask is not None:
            # Get the last non-padded token for each sequence
            batch_size = input_ids.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            pooled_output = last_hidden_states[range(batch_size), sequence_lengths]
        else:
            # Use the last token
            pooled_output = last_hidden_states[:, -1, :]  # [batch_size, hidden_size]

        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]

        return logits

    @classmethod
    def from_pretrained(cls, checkpoint_path, **kwargs):
        """Load model from checkpoint."""
        if isinstance(checkpoint_path, str) and checkpoint_path.endswith(".pt"):
            # Load from checkpoint file
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Get model parameters from checkpoint
            model_name = checkpoint.get("model_name", "M4-ai/TinyMistral-6x248M")
            num_classes = checkpoint.get("num_classes", 4)

            # Create model
            model = cls(model_name=model_name, num_classes=num_classes, **kwargs)

            # Load state dict
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            logger.info(f"✅ Model loaded from checkpoint: {checkpoint_path}")
            return model
        else:
            # Create new model
            return cls(**kwargs)

    def save_pretrained(self, save_path):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "config": self.model.config,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"✅ Model saved to: {save_path}")


def load_tinymistral_tokenizer(model_name="M4-ai/TinyMistral-6x248M"):
    """Load TinyMistral tokenizer."""
    logger.info(f"Loading TinyMistral tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("✅ TinyMistral tokenizer loaded")
    return tokenizer
