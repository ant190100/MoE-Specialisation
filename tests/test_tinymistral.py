"""
Test TinyMistral model integration.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.tinymistral_classifier import TinyMistralForClassification, load_tinymistral_tokenizer

def test_tinymistral_loading():
    """Test if TinyMistral can be loaded and run."""
    print("🧪 Testing TinyMistral model loading...")
    print("⏳ Note: First download may take several minutes...")
    
    try:
        # Test tokenizer loading
        print("\n1️⃣ Loading tokenizer...")
        tokenizer = load_tinymistral_tokenizer()
        print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Pad token: {tokenizer.pad_token}")
        
        # Test model loading
        print("\n2️⃣ Loading model (this will take a while)...")
        model = TinyMistralForClassification(num_classes=4)
        print(f"✅ Model loaded successfully!")
        
        # Test forward pass
        print("\n3️⃣ Testing forward pass...")
        test_text = "This is a test sentence for classification."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        print(f"   Input shape: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs.get("attention_mask"))
            
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Sample logits: {outputs[0].tolist()}")
        
        # Test model info
        print("\n4️⃣ Model information:")
        model_info = model.get_expert_info()
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        print("\n🎉 TinyMistral test completed successfully!")
        print("💡 Ready to run full experiment!")
        return True
        
    except Exception as e:
        print(f"\n❌ TinyMistral test failed: {e}")
        print("💡 This might be due to:")
        print("   - Network connectivity issues")
        print("   - Insufficient disk space")
        print("   - Missing dependencies")
        print("   - Memory constraints")
        return False

def main():
    """Main test function."""
    print("="*60)
    print("🤗 TINYMISTRAL MODEL TEST")
    print("="*60)
    
    success = test_tinymistral_loading()
    
    print("\n" + "="*60)
    if success:
        print("✅ TinyMistral is ready for experiments!")
        print("\nNext steps:")
        print("🚀 Run full experiment: python experiments/tinymistral_experiment.py")
        print("📊 Compare with toy model: python analysis_scripts/local_analysis.py --compare")
    else:
        print("❌ TinyMistral setup needs fixing.")
        print("\nTroubleshooting:")
        print("🔧 Check internet connection")
        print("💾 Ensure sufficient disk space (~2GB)")
        print("🐍 Update transformers: pip install transformers --upgrade")
    print("="*60)

if __name__ == "__main__":
    main()
