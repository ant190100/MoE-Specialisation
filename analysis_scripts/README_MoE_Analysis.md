# TinyMistral MoE Analysis Scripts

## Overview
Created three specialized analysis scripts for the TinyMistral-6x248M Mixture of Experts model:

## Scripts Created

### 1. Routing Entropy Analysis (`tinymistral_routing_entropy.py`) - Simplified
**Purpose**: Analyze how uncertain the routing decisions are across different layers and tokens.

**Key Features**:
- **Entropy by Layer**: Shows mean entropy progression across all 12 MoE layers with standard deviation
- **Single Layer Distribution**: Detailed histogram of entropy distribution for a specified target layer
- **Configurable Target Layer**: Use `--target-layer` parameter to specify which layer to analyze in detail
- **Statistical Summary**: Layer-wise entropy statistics and target layer details

**Key Findings**:
- Average entropy: 0.968 bits across all layers
- Entropy decreases from early layers (1.305 bits) to later layers (0.693 bits)
- This suggests the model becomes more confident in expert selection as depth increases
- Range: 0.028 - 1.712 bits showing diverse routing patterns

**Usage**:
```bash
# Analyze with default target layer 6
python analysis_scripts/tinymistral_routing_entropy.py

# Analyze with specific target layer (e.g., layer 11)
python analysis_scripts/tinymistral_routing_entropy.py --target-layer 11
```

### 2. Load Balancing Analysis (`tinymistral_load_balancing.py`) - Simplified
**Purpose**: Analyze how evenly tokens are distributed across the 6 experts in each layer.

**Key Features**:
- **Expert Usage Tracking**: Monitors which experts handle which tokens
- **Load Distribution Visualization**: Shows usage percentage per expert for each layer
- **Balance Quality Metrics**: Coefficient of Variation to measure balance quality
- **Perfect Balance Reference**: Red dashed line showing ideal 16.7% usage per expert

**Key Findings**:
- Poor load balancing across all layers (CV > 1.0)
- Most extreme imbalance in Layer 11 (CV=1.432)
- Expert 2 in Layer 11 handles 61.4% of tokens (severe imbalance)
- Some experts in early layers receive 0% of tokens

**Output**: 
- Single visualization with 12 subplots (one per layer) showing expert usage distribution
- CSV file with detailed usage statistics
- Console summary with balance quality metrics

### 3. Expert Ablation Analysis (`tinymistral_simple_accuracy_ablation.py`) - Lightweight
**Purpose**: Determine the importance of individual experts by measuring accuracy degradation when experts are disabled.

**Key Features**:
- **Expert Ablation**: Systematically zeros out individual expert weights and measures performance impact
- **Next-Token Prediction**: Uses lightweight next-token prediction accuracy instead of full classification
- **Category-Specific Analysis**: Measures accuracy drops across different news categories (World, Sports, Business, Sci/Tech)
- **Expert Ranking**: Identifies which experts contribute most to overall model performance
- **Single Layer Focus**: Analyzes one specified layer at a time for efficiency

**Key Findings**:
- Identifies most/least impactful experts within a layer
- Shows category-specific expert specialization patterns  
- Reveals expert redundancy through minimal accuracy drops
- Demonstrates performance degradation when critical experts are removed

**Usage**:
```bash
# Analyze layer 6 with default 80 samples
python analysis_scripts/tinymistral_simple_accuracy_ablation.py --single-layer 6

# Analyze layer 0 with more samples for higher precision
python analysis_scripts/tinymistral_simple_accuracy_ablation.py --single-layer 0 --samples 200

# Custom output directory
python analysis_scripts/tinymistral_simple_accuracy_ablation.py --single-layer 11 --output-dir results/my_ablation
```

**Output**: 
- Accuracy change heatmap (baseline vs ablated experts)
- Expert impact ranking (total accuracy drop)
- CSV files with detailed results and impact summaries

## Technical Implementation

### Architecture Integration
- **Model Compatibility**: Specifically designed for TinyMistral-6x248M architecture
- **Hook-based Extraction**: Uses forward hooks on `block_sparse_moe.gate` modules
- **Tensor Handling**: Robust handling of different tensor shapes (batch vs single sequence)
- **Memory Efficient**: Detaches tensors and uses CPU storage for large analyses

### Routing Analysis Details
- **Router Logits**: Extracts raw router outputs before softmax
- **Top-k Analysis**: Handles the top-2 routing mechanism
- **Probability Distributions**: Analyzes softmax probabilities for entropy calculation
- **Token Masking**: Only analyzes valid tokens using attention masks

### Ablation Analysis Details
- **Expert Zeroing**: Temporarily sets all expert weights to zero to measure impact
- **Next-Token Prediction**: Uses lightweight token completion tasks instead of full classification
- **Category-Specific Metrics**: Measures performance across different text types (World, Sports, Business, Sci/Tech)
- **Restoration**: Automatically restores original weights after each ablation test
- **Single Layer Focus**: Analyzes one layer at a time for computational efficiency

## Usage Examples

### Routing Entropy Analysis
```bash
cd /path/to/MoE-Specialisation

# Default analysis (target layer 6)
python analysis_scripts/tinymistral_routing_entropy.py

# Analyze specific layer (e.g., final layer 11)
python analysis_scripts/tinymistral_routing_entropy.py --target-layer 11

# Custom parameters
python analysis_scripts/tinymistral_routing_entropy.py --target-layer 3 --batch-size 16
```

### Load Balancing Analysis  
```bash
cd /path/to/MoE-Specialisation
python analysis_scripts/tinymistral_load_balancing.py
```

### Expert Ablation Analysis
```bash
cd /path/to/MoE-Specialisation

# Analyze specific layer (required parameter)
python analysis_scripts/tinymistral_simple_accuracy_ablation.py --single-layer 6

# Higher precision with more samples
python analysis_scripts/tinymistral_simple_accuracy_ablation.py --single-layer 0 --samples 200
```

**Output**:
- **Entropy Analysis**: 
  - Mean entropy by layer (line plot with ±1 std deviation)
  - Target layer entropy distribution (histogram with mean/median lines)
- **Load Balancing**: Single load distribution plot with 12 layer subplots, usage statistics
- **Ablation Analysis**: 
  - Accuracy change heatmap (baseline vs expert ablation)
  - Expert impact ranking showing most critical experts

## Results Location
- **Entropy Analysis**: `results/tinymistral_entropy/`
- **Load Balancing**: `results/tinymistral_load_balancing/`
- **Ablation Analysis**: `results/tinymistral_simple_accuracy/`

## Key Insights

### Model Behavior
1. **Routing Confidence**: Model becomes more confident (lower entropy) in deeper layers
2. **Load Imbalance**: Severe load balancing issues with some experts heavily overused
3. **Expert Specialization**: Evidence of expert specialization but at the cost of balance
4. **Expert Importance**: Ablation reveals varying levels of expert contribution to model performance
5. **Layer Patterns**: Different routing patterns and expert importance emerge at different depths

### Implications for MoE Training
1. **Load Balancing Loss**: Current model would benefit from stronger load balancing regularization
2. **Expert Utilization**: Many experts are underutilized, suggesting inefficient capacity usage
3. **Routing Quality**: Lower entropy in deeper layers suggests good routing convergence
4. **Critical Expert Identification**: Ablation helps identify which experts are essential vs redundant
5. **Scaling Considerations**: Load balancing becomes worse in deeper layers, but expert importance varies

## Compatibility
- ✅ **TinyMistral-6x248M**: Fully compatible with the trained model
- ✅ **AG News Dataset**: Integrated with the classification task
- ✅ **Custom Model Loading**: Works with locally saved model checkpoints
- ✅ **Batch Processing**: Handles variable batch sizes and sequence lengths

## Comprehensive Analysis Workflow

These three scripts provide complementary insights into MoE model behavior:

1. **Routing Entropy** → Measures routing uncertainty and confidence patterns
2. **Load Balancing** → Identifies usage imbalances across experts  
3. **Expert Ablation** → Determines functional importance of individual experts

**Recommended Analysis Order**:
1. Start with **Load Balancing** to understand overall expert utilization
2. Use **Routing Entropy** to analyze routing confidence patterns
3. Apply **Expert Ablation** to identify critical vs redundant experts

**Combined Insights**: Together, these analyses reveal whether load imbalances correspond to actual importance differences, helping distinguish between underutilized-but-critical vs truly redundant experts.

## Troubleshooting Notes
- **Tensor Shape Issues**: Fixed dimension mismatches in router hook extraction
- **Memory Management**: Uses CPU storage for large analyses to avoid GPU memory issues  
- **Hook Registration**: Proper cleanup of forward hooks to prevent memory leaks
- **Error Handling**: Robust handling of different model architectures and tensor shapes
