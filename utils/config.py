"""
Configuration management utilities.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def setup_experiment_paths(experiment_name, base_results_dir="./results"):
    """Setup directory structure for an experiment."""
    base_path = Path(base_results_dir) / experiment_name
    paths = {
        "experiment_dir": base_path,
        "models_dir": base_path / "models",
        "logs_dir": base_path / "logs",
        "analysis_dir": base_path / "analysis",
        "configs_dir": base_path / "configs",
    }

    # Create directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def merge_configs(base_config, override_config):
    """Merge two configuration dictionaries."""
    merged = base_config.copy()
    merged.update(override_config)
    return merged
