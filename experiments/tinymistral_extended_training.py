"""
Extended training script for TinyMistral with more epochs and advanced features.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.tinymistral_classifier import TinyMistralForClassification, load_tinymistral_tokenizer
from src.data.loaders import create_data_loaders

class TinyMistralExtendedTraining:
    """Extended training for TinyMistral with proper validation and checkpointing."""
    
    def __init__(self, config_path="configs/tinymistral_training.yaml"):
        # Load training configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.setup_device()
        self.setup_paths()
        self.setup_logging()
        
    def setup_device(self):
        """Setup the best available device."""
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"🎯 Using CUDA: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            print("🍎 Using Apple Silicon MPS")
        else:
            self.device = "cpu"
            print("💻 Using CPU")
            
    def setup_paths(self):
        """Setup experiment directory structure."""
        self.results_dir = Path("results") / "tinymistral_extended_training"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "analysis").mkdir(exist_ok=True)
        (self.results_dir / "checkpoints").mkdir(exist_ok=True)
        
        print(f"📁 Results directory: {self.results_dir}")
        
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_file = (self.results_dir / "logs" / 
                   f"extended_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
    def setup_model_and_data(self):
        """Initialize model, tokenizer, and data loaders."""
        self.logger.info("🤖 Setting up model and data...")
        
        # Load tokenizer and model
        print("📝 Loading TinyMistral tokenizer...")
        self.tokenizer = load_tinymistral_tokenizer(self.config["MODEL_NAME"])
        
        print("🔄 Loading TinyMistral model...")
        self.model = TinyMistralForClassification(
            model_name=self.config["MODEL_NAME"],
            num_classes=self.config["NUM_CLASSES"]
        ).to(self.device)
        
        # Create data loaders
        print("📚 Creating data loaders...")
        self.train_loader, self.val_loader, self.class_names = create_data_loaders(
            dataset_name=self.config["DATASET"],
            batch_size=self.config["BATCH_SIZE"],
            tokenizer=self.tokenizer,
        )
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        
        # CRITICAL: Only train classification head, freeze base model
        print("❄️ Freezing base model parameters...")
        trainable_params = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"📊 Trainable parameters: {trainable_params:,}")
        print(f"❄️ Frozen parameters: {frozen_params:,}")
        print(f"📈 Training ratio: {trainable_params/(trainable_params+frozen_params)*100:.4f}%")
        
        # Setup optimizer - only trainable parameters
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config["LEARNING_RATE"],
            weight_decay=self.config["WEIGHT_DECAY"]
        )
        
        # Learning rate scheduler
        if self.config.get("USE_SCHEDULER", False):
            if self.config.get("SCHEDULER_TYPE") == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.config["NUM_EPOCHS"]
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=5, gamma=0.5
                )
        else:
            self.scheduler = None
        
        self.logger.info("✅ Model and data setup complete!")
        
    def train_epoch(self, epoch):
        """Train for one epoch with detailed monitoring."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Determine number of batches
        max_batches = min(self.config["MAX_BATCHES_PER_EPOCH"], len(self.train_loader))
        
        progress_bar = tqdm(
            enumerate(self.train_loader), 
            total=max_batches,
            desc=f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}"
        )
        
        for i, batch in progress_bar:
            if i >= max_batches:
                break
                
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get("GRADIENT_CLIPPING", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["GRADIENT_CLIPPING"]
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            current_accuracy = 100.0 * correct_predictions / total_predictions
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}',
                'Acc': f'{current_accuracy:.2f}%'
            })
            
            # Detailed logging
            if self.config.get("DETAILED_LOGGING", False) and i % self.config.get("LOG_EVERY_N_BATCHES", 50) == 0:
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {i:3d}: "
                    f"Loss={loss.item():.4f}, "
                    f"Acc={current_accuracy:.2f}%, "
                    f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # Apply scheduler
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        train_accuracy = 100.0 * correct_predictions / total_predictions
        
        self.logger.info(
            f"Epoch {epoch+1} Summary - "
            f"Loss: {avg_loss:.4f}, "
            f"Train Acc: {train_accuracy:.2f}%, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        return avg_loss, train_accuracy
        
    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        num_batches = 0
        class_correct = {i: 0 for i in range(self.config["NUM_CLASSES"])}
        class_total = {i: 0 for i in range(self.config["NUM_CLASSES"])}
        
        # Determine validation batch limit based on device (much more aggressive for CPU)
        max_val_batches = self.config.get("MAX_VAL_BATCHES", 50)  # Default to 50
        if self.device == "cpu":
            max_val_batches = min(max_val_batches, 20)  # Limit to 20 batches on CPU
            print(f"🎯 Running validation (CPU optimized: {max_val_batches} batches)...")
        else:
            print(f"🎯 Running validation ({max_val_batches} batches)...")
        
        # Add progress bar for validation
        val_progress = tqdm(
            enumerate(self.val_loader), 
            total=max_val_batches,
            desc="Validation",
            leave=False
        )
        
        start_time = time.time()
        
        with torch.no_grad():
            for i, batch in val_progress:
                if i >= max_val_batches:
                    break
                    
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                num_batches += 1
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar with current stats
                current_acc = 100 * correct / total if total > 0 else 0
                val_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%',
                    'Samples': f'{total}'
                })
                
                # Per-class accuracy
                for j in range(labels.size(0)):
                    label = labels[j].item()
                    class_total[label] += 1
                    if predicted[j] == labels[j]:
                        class_correct[label] += 1
        
        val_time = time.time() - start_time            
        accuracy = 100 * correct / total
        avg_val_loss = val_loss / num_batches
        
        # Per-class accuracy logging
        class_accuracies = {}
        for i in range(self.config["NUM_CLASSES"]):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                class_accuracies[self.class_names[i]] = class_acc
        
        print(f"✅ Validation completed in {val_time:.1f}s ({num_batches} batches, {total} samples)")        
        self.logger.info(f"Validation - Loss: {avg_val_loss:.4f}, Overall Acc: {accuracy:.2f}% ({val_time:.1f}s)")
        
        if self.config.get("DETAILED_LOGGING", False):
            for class_name, class_acc in class_accuracies.items():
                self.logger.info(f"  {class_name}: {class_acc:.2f}%")
        
        return accuracy, avg_val_loss, class_accuracies
        
    def save_checkpoint(self, epoch, accuracy, train_losses, val_accuracies, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': accuracy,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        if is_best:
            save_path = self.results_dir / "models" / "best_model.pt"
            torch.save(checkpoint, save_path)
            self.logger.info(f"💾 Best model saved to {save_path}")
        
        # Regular checkpoint
        checkpoint_path = self.results_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
        
    def train(self):
        """Full training loop with validation and checkpointing."""
        self.logger.info("🚀 Starting extended training...")
        self.logger.info(f"📊 Configuration: {self.config}")
        
        # Training history
        best_accuracy = 0
        patience_counter = 0
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        val_losses = []
        
        print(f"\n🎯 Training for {self.config['NUM_EPOCHS']} epochs")
        print(f"📚 Dataset: {self.config['DATASET']}")
        print(f"🔢 Batch size: {self.config['BATCH_SIZE']}")
        print(f"📈 Learning rate: {self.config['LEARNING_RATE']}")
        print(f"⏰ Max batches per epoch: {self.config['MAX_BATCHES_PER_EPOCH']}")
        
        for epoch in range(self.config["NUM_EPOCHS"]):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{self.config['NUM_EPOCHS']}")
            print(f"{'='*60}")
            
            # Train epoch
            train_loss, train_acc = self.train_epoch(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Evaluate
            if (epoch + 1) % self.config["EVAL_EVERY_N_EPOCHS"] == 0:
                val_accuracy, val_loss, class_accuracies = self.evaluate()
                val_accuracies.append(val_accuracy)
                val_losses.append(val_loss)
                
                # Check for improvement
                improvement = val_accuracy - best_accuracy
                if improvement > self.config["MIN_DELTA"]:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        epoch, val_accuracy, train_losses, val_accuracies, is_best=True
                    )
                    
                    print(f"🏆 NEW BEST MODEL! Accuracy: {best_accuracy:.2f}% (+{improvement:.2f}%)")
                else:
                    patience_counter += 1
                    print(f"📉 No improvement (patience: {patience_counter}/{self.config['PATIENCE']})")
                    
                # Early stopping
                if patience_counter >= self.config["PATIENCE"]:
                    print(f"⏹️ Early stopping triggered after {epoch+1} epochs")
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config["SAVE_EVERY_N_EPOCHS"] == 0:
                checkpoint_path = self.save_checkpoint(
                    epoch, best_accuracy, train_losses, val_accuracies
                )
                print(f"💾 Checkpoint saved: {checkpoint_path.name}")
                
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"🏆 Best validation accuracy: {best_accuracy:.2f}%")
        self.logger.info(f"Training complete! Best accuracy: {best_accuracy:.2f}%")
        
        # Save final training history
        training_history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'val_losses': val_losses,
            'best_accuracy': best_accuracy,
            'config': self.config,
            'class_names': self.class_names
        }
        
        history_path = self.results_dir / "analysis" / "training_history.pt"
        torch.save(training_history, history_path)
        self.logger.info(f"Training history saved to {history_path}")
        
        return best_accuracy
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint and return training state."""
        self.logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if hasattr(self, 'scheduler') and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        accuracy = checkpoint['best_accuracy']
        train_losses = checkpoint.get('train_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        
        self.logger.info(f"✅ Loaded checkpoint from epoch {epoch+1}, accuracy: {accuracy:.2f}%")
        return epoch, accuracy, train_losses, val_accuracies

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint file."""
        # Look for actual model checkpoints, not training history
        checkpoint_patterns = [
            self.results_dir / "models" / "best_model.pt",
            self.results_dir / "checkpoints" / "checkpoint_epoch_3.pt",
            self.results_dir / "checkpoints" / "checkpoint_epoch_2.pt",
            self.results_dir / "checkpoints" / "checkpoint_epoch_1.pt"
        ]
        
        # Check specific patterns first
        for pattern in checkpoint_patterns:
            if pattern.exists():
                print(f"🎯 Found checkpoint: {pattern.relative_to(self.results_dir)}")
                return pattern
        
        # Search for any .pt files that are likely checkpoints (exclude training_history)
        all_checkpoints = []
        for pt_file in self.results_dir.rglob("*.pt"):
            if "training_history" not in str(pt_file) and "results" not in str(pt_file):
                all_checkpoints.append(pt_file)
                
        if all_checkpoints:
            # Sort by modification time, most recent first
            all_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            print(f"🔍 Found {len(all_checkpoints)} checkpoint files:")
            for i, cp in enumerate(all_checkpoints[:5]):  # Show top 5
                size_mb = cp.stat().st_size / (1024 * 1024)
                modified = datetime.fromtimestamp(cp.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"   {i+1}. {cp.relative_to(self.results_dir)} ({size_mb:.1f}MB, {modified})")
            
            return all_checkpoints[0]  # Return most recent
        
        return None

    def load_from_checkpoint(self, checkpoint_path=None):
        """Setup model and load from checkpoint without training components."""
        self.logger.info("🤖 Setting up model for inference...")
        
        # Load tokenizer and model
        print("📝 Loading TinyMistral tokenizer...")
        self.tokenizer = load_tinymistral_tokenizer(self.config["MODEL_NAME"])
        
        print("🔄 Loading TinyMistral model...")
        self.model = TinyMistralForClassification(
            model_name=self.config["MODEL_NAME"],
            num_classes=self.config["NUM_CLASSES"]
        ).to(self.device)
        
        # Create data loaders
        print("📚 Creating data loaders...")
        self.train_loader, self.val_loader, self.class_names = create_data_loaders(
            dataset_name=self.config["DATASET"],
            batch_size=self.config["BATCH_SIZE"],
            tokenizer=self.tokenizer,
        )
        
        # Setup criterion for evaluation
        self.criterion = nn.CrossEntropyLoss()
        
        # Load checkpoint if provided or find latest
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
            
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found! Run training first.")
        
        print(f"📂 Using checkpoint: {checkpoint_path.relative_to(self.results_dir)}")
        epoch, accuracy, train_losses, val_accuracies = self.load_checkpoint(checkpoint_path)
        
        print(f"✅ Loaded trained model from epoch {epoch+1} with {accuracy:.2f}% accuracy")
        return epoch, accuracy, train_losses, val_accuracies
        
    def plot_training_curves(self):
        """Create comprehensive training visualizations."""
        history_path = self.results_dir / "analysis" / "training_history.pt"
        if not history_path.exists():
            self.logger.warning("No training history found for plotting")
            return
            
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
            return
            
        history = torch.load(history_path)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        ax1.plot(history['train_losses'], label='Training Loss', color='blue', linewidth=2)
        ax1.set_title('Training Loss Over Time', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Training accuracy
        ax2.plot(history['train_accuracies'], label='Training Accuracy', color='green', linewidth=2)
        ax2.set_title('Training Accuracy Over Time', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Validation accuracy
        eval_epochs = list(range(
            self.config["EVAL_EVERY_N_EPOCHS"]-1, 
            len(history['val_accuracies']) * self.config["EVAL_EVERY_N_EPOCHS"], 
            self.config["EVAL_EVERY_N_EPOCHS"]
        ))
        ax3.plot(eval_epochs, history['val_accuracies'], 
                label='Validation Accuracy', color='orange', linewidth=2, marker='o')
        ax3.set_title('Validation Accuracy Over Time', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Combined view
        ax4.plot(history['train_losses'], label='Train Loss', color='blue', alpha=0.7)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(eval_epochs, history['val_accuracies'], 
                     label='Val Accuracy', color='red', linewidth=2, marker='s')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='blue')
        ax4_twin.set_ylabel('Accuracy (%)', color='red')
        ax4.set_title('Training Overview', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add best accuracy annotation
        if history['val_accuracies']:
            best_idx = np.argmax(history['val_accuracies'])
            best_epoch = eval_epochs[best_idx]
            best_acc = history['val_accuracies'][best_idx]
            ax4_twin.annotate(f'Best: {best_acc:.2f}%', 
                            xy=(best_epoch, best_acc), 
                            xytext=(best_epoch+2, best_acc+5),
                            arrowprops=dict(arrowstyle='->', color='red'),
                            fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "analysis" / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training curves saved to {plot_path}")
        self.logger.info(f"Training curves saved to {plot_path}")
        
    def run(self):
        """Run the complete extended training pipeline."""
        try:
            # Setup
            self.setup_model_and_data()
            
            # Train
            best_accuracy = self.train()
            
            # Visualize
            if self.config.get("SAVE_TRAINING_CURVES", True):
                self.plot_training_curves()
            
            # Final summary
            print(f"\n{'='*60}")
            print("🎉 EXTENDED TRAINING COMPLETED!")
            print(f"{'='*60}")
            print(f"🏆 Best accuracy: {best_accuracy:.2f}%")
            print(f"📁 Results saved in: {self.results_dir}")
            print(f"📊 Models saved in: {self.results_dir}/models/")
            print(f"📈 Training curves: {self.results_dir}/analysis/")
            print(f"📝 Logs: {self.results_dir}/logs/")
            
            self.logger.info("✅ Extended training pipeline completed successfully!")
            return best_accuracy
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            print(f"❌ Training failed: {e}")
            raise

if __name__ == "__main__":
    trainer = TinyMistralExtendedTraining()
    trainer.run()
