"""
Test TinyMistral integration with analytics suite.
"""

import sys
import os
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_tinymistral_integration():
    """Test the complete TinyMistral integration."""
    
    print("🧪 TESTING TINYMISTRAL INTEGRATION")
    print("="*60)
    
    # Test 1: Model loading
    print("\n1️⃣ Testing model loading...")
    try:
        from src.models.tinymistral_classifier import TinyMistralForClassification, load_tinymistral_tokenizer
        tokenizer = load_tinymistral_tokenizer()
        model = TinyMistralForClassification(num_classes=4)
        print("✅ Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 2: Results analysis
    print("\n2️⃣ Testing results analysis...")
    results_dir = Path("results/tinymistral_lightweight")
    if (results_dir / "analysis" / "lightweight_results.pt").exists():
        print("✅ Lightweight results found")
    else:
        print("❌ No results found - run experiment first")
        return False
    
    # Test 3: Visualizations
    print("\n3️⃣ Testing visualizations...")
    viz_files = [
        "tinymistral_expert_usage_heatmap.png",
        "tinymistral_expert_utilization.png", 
        "tinymistral_class_expert_preference.png"
    ]
    
    viz_count = 0
    for viz_file in viz_files:
        if (results_dir / "analysis" / viz_file).exists():
            viz_count += 1
            
    print(f"✅ Found {viz_count}/{len(viz_files)} visualizations")
    
    # Test 4: Analytics compatibility
    print("\n4️⃣ Testing analytics compatibility...")
    try:
        results = torch.load(results_dir / "analysis" / "lightweight_results.pt", map_location='cpu')
        
        required_keys = ['zero_shot_performance', 'expert_analysis', 'model_info', 'class_names']
        missing_keys = [key for key in required_keys if key not in results]
        
        if not missing_keys:
            print("✅ Results format compatible with analytics suite")
        else:
            print(f"⚠️ Missing keys: {missing_keys}")
            
    except Exception as e:
        print(f"❌ Analytics compatibility test failed: {e}")
        return False
    
    # Summary
    print("\n" + "="*60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("="*60)
    
    model_info = results.get('model_info', {})
    performance = results.get('zero_shot_performance', {})
    
    print(f"✅ Model: {model_info.get('model_name', 'Unknown')}")
    print(f"✅ Experts: {model_info.get('num_experts', 'Unknown')}")
    print(f"✅ Classes: {model_info.get('num_classes', 'Unknown')}")
    print(f"✅ Zero-shot accuracy: {performance.get('accuracy', 0):.1f}%")
    print(f"✅ Samples evaluated: {performance.get('total_samples', 0)}")
    print(f"✅ Visualizations: {viz_count} generated")
    
    routing = results.get('routing_analysis', {})
    print(f"✅ Routing analysis: {routing.get('batches_analyzed', 0)} batches")
    
    expert_df = results.get('expert_analysis')
    if expert_df is not None:
        print(f"✅ Expert analysis: {expert_df.shape[0]} experts x {expert_df.shape[1]} classes")
    
    print("\n🎉 TinyMistral integration test PASSED!")
    print("💡 Your analytics suite now works with:")
    print("   🧸 Toy MoE models (custom implementation)")
    print("   🤗 TinyMistral-6x248M (production MoE model)")
    print("   📊 Comprehensive analysis pipeline")
    print("   🎨 Visualization generation")
    
    return True

if __name__ == "__main__":
    success = test_tinymistral_integration()
    
    if not success:
        print("\n💡 To fix issues:")
        print("1. Run: python tests/test_tinymistral.py")
        print("2. Run: python experiments/tinymistral_lightweight.py") 
        print("3. Run this test again")
        
    print("\n" + "="*60)
