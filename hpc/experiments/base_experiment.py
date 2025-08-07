"""
Base experiment class for HPC runs.
"""

import os
import yaml
import torch
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


class BaseExperiment(ABC):
    """Base class for all HPC experiments."""

    def __init__(self, config_path, experiment_name=None):
        self.config = self.load_config(config_path)
        self.experiment_name = experiment_name or self.config.get(
            "EXPERIMENT_NAME", "experiment"
        )
        self.setup_paths()
        self.setup_logging()

    def load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def setup_paths(self):
        """Setup experiment directory structure."""
        base_dir = (
            Path(self.config.get("HPC_RESULTS_DIR", "./results")) / self.experiment_name
        )

        self.paths = {
            "experiment_dir": base_dir,
            "models_dir": base_dir / "models",
            "logs_dir": base_dir / "logs",
            "analysis_dir": base_dir / "analysis",
            "configs_dir": base_dir / "configs",
        }

        # Create directories
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging for the experiment."""
        log_file = (
            self.paths["logs_dir"]
            / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    def save_results(self, results, filename):
        """Save experiment results."""
        output_path = self.paths["models_dir"] / filename
        torch.save(results, output_path)
        self.logger.info(f"Results saved to {output_path}")

    def save_config(self):
        """Save the current configuration."""
        config_path = self.paths["configs_dir"] / f"{self.experiment_name}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self.logger.info(f"Configuration saved to {config_path}")

    @abstractmethod
    def run(self):
        """Run the experiment. Must be implemented by subclasses."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.save_config()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.logger.error(f"Experiment failed with error: {exc_val}")
        else:
            self.logger.info(f"Experiment completed: {self.experiment_name}")
        return False
