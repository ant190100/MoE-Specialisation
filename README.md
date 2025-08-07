# MoE Specialisation Research

A research repository for analyzing expert specialization in Mixture of Experts (MoE) models, designed for both local development and HPC cluster execution.

**Repository Status**: ✅ **Clean & Optimized** - All redundancies removed, structure validated, tests passing.

## Repository Structure

```
MoE-Specialisation/
├── src/                          # Core source code
│   ├── models/                   # Model implementations
│   │   ├── expert.py            # Individual expert networks
│   │   ├── moe_layer.py         # MoE layer implementation
│   │   └── tiny_moe.py          # Complete TinyMoE model
│   ├── data/                    # Data handling utilities
│   │   └── loaders.py           # Dataset loaders and collate functions
│   ├── analysis/                # Analysis tools
│   │   ├── ablation.py          # Ablation analysis
│   │   └── visualization.py     # Plotting and visualization
│   └── utils/                   # Utility functions
│       └── config.py            # Configuration management
├── experiments/                 # Local experiment scripts
│   └── toy_model_experiment.py  # Toy model training and analysis
├── hpc/                         # HPC-specific code
│   ├── experiments/             # Experiment scripts for cluster
│   │   ├── base_experiment.py   # Base experiment class
│   │   └── tinymistral_experiment.py
│   ├── job_scripts/             # SLURM job scripts
│   │   └── tinymistral_analysis.slurm
│   └── configs/                 # HPC-specific configurations
│       └── tinymistral_config.yaml
├── analysis_scripts/            # Local analysis tools
│   ├── local_analysis.py        # Process HPC results locally
│   └── sync_hpc.py             # Sync files with HPC
├── tests/                       # Testing infrastructure
│   ├── test_local.py           # Comprehensive local testing
│   └── test_structure.py       # Repository structure validation
├── notebooks/                   # Jupyter notebooks for exploration
│   └── toy_model_workflow.ipynb # Original workflow notebook
├── results/                     # Results storage (git-ignored)
│   └── {experiment_name}/       # Individual experiment results
│       ├── models/              # Trained model weights
│       ├── analysis/            # Analysis outputs and visualizations
│       ├── logs/                # Execution logs
│       └── configs/             # Experiment configurations
├── configs/                     # Configuration files
│   └── toy_model_config.yaml   # Local toy model config
└── requirements/                # Dependencies
    ├── local.txt               # Local development deps
    └── hpc.txt                 # HPC cluster deps
```

## Workflow

### 1. Local Development & Testing

```bash
# Install dependencies
pip install -r requirements/local.txt

# Quick local test with comprehensive validation
python tests/test_local.py

# Validate repository structure
python tests/test_structure.py

# Work with notebooks
jupyter lab notebooks/
```

### 2. HPC Execution

```bash
# Submit TinyMistral analysis
sbatch hpc/job_scripts/tinymistral_analysis.slurm

# Check job status
squeue -u $USER
```

### 3. Local Analysis of HPC Results

```bash
# Sync results from HPC
python analysis_scripts/sync_hpc.py --direction from-hpc --host your-cluster.edu --remote-path /scratch/user/results

# Analyze toy model results (local experiments)
python analysis_scripts/local_analysis.py --experiment results/toy_model_analysis --type toy_model

# Analyze TinyMistral results (HPC experiments)  
python analysis_scripts/local_analysis.py --experiment results/tinymistral_analysis --type tinymistral

# Compare multiple experiments
python analysis_scripts/local_analysis.py --compare results/toy_model_analysis results/tinymistral_analysis
```

## Configuration

### Toy Model Config (`configs/toy_model_config.yaml`)
```yaml
EMBED_DIM: 32
HIDDEN_DIM: 64
NUM_EXPERTS: 4
TOP_K: 2
NUM_CLASSES: 4
DATASET: "ag_news"
```

### HPC Config (`hpc/configs/tinymistral_config.yaml`)
```yaml
MODEL_PATH: "/scratch/user/models/TinyMistral-6x248M"
NUM_EXPERTS: 6
NUM_CLASSES: 20
DATASET: "20_newsgroups"
HPC_RESULTS_DIR: "/scratch/user/results"
```

## Key Features

### 🔬 **Ablation Analysis**
- Systematic expert ablation to measure specialization
- Per-class accuracy impact measurement
- Statistical analysis and visualization

### 📊 **Visualizations**
- Expert specialization heatmaps
- Routing entropy distributions
- Expert utilization statistics
- Confidence analysis plots

### 🖥️ **HPC Integration**
- SLURM job scripts for cluster execution
- Experiment tracking and logging
- Result synchronization utilities
- Configurable resource requirements

### 🔧 **Modular Design**
- Reusable model components
- Configurable experiments
- Extensible analysis framework
- Clean separation of concerns

## Getting Started

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd MoE-Specialisation
   pip install -e .
   ```

2. **Local testing:**
   ```bash
   # Run comprehensive tests
   python tests/test_local.py
   
   # Validate structure
   python tests/test_structure.py
   
   # Run local toy model experiment
   python experiments/toy_model_experiment.py
   ```

3. **HPC usage:**
   - Modify `hpc/configs/` for your cluster
   - Update SLURM scripts with your paths
   - Submit jobs and sync results

4. **Analysis:**
   ```bash
   python analysis_scripts/local_analysis.py --help
   ```

## Research Applications

This framework supports research into:
- **Expert Specialization**: How do experts specialize across different domains?
- **Routing Patterns**: What drives expert selection in MoE models?
- **Scalability**: How does specialization change with model size?
- **Transfer Learning**: Do specialization patterns transfer across tasks?

## Models Supported

- **TinyMoE**: Custom lightweight MoE for quick experimentation
- **TinyMistral**: Pre-trained MoE model analysis
- **Extensible**: Easy to add new MoE architectures

## Contributing

1. Add new models in `src/models/`
2. Extend analysis in `src/analysis/`
3. Create experiment scripts in `hpc/experiments/`
4. Add visualization in `src/analysis/visualization.py`

## License

MIT License