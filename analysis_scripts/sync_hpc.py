"""
Utility script for syncing results from HPC to local machine.
"""

import subprocess
import argparse
from pathlib import Path
import yaml


def sync_from_hpc(hpc_host, remote_path, local_path, dry_run=False):
    """Sync results from HPC to local machine."""
    cmd = ["rsync", "-avz", "--progress", f"{hpc_host}:{remote_path}/", str(local_path)]

    if dry_run:
        cmd.append("--dry-run")
        print("DRY RUN - would execute:")

    print(" ".join(cmd))

    if not dry_run:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Sync completed successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Sync failed: {e}")
            print(e.stderr)


def sync_to_hpc(hpc_host, local_path, remote_path, dry_run=False):
    """Sync local files to HPC."""
    cmd = [
        "rsync",
        "-avz",
        "--progress",
        f"{local_path}/",
        f"{hpc_host}:{remote_path}/",
    ]

    if dry_run:
        cmd.append("--dry-run")
        print("DRY RUN - would execute:")

    print(" ".join(cmd))

    if not dry_run:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Sync completed successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Sync failed: {e}")
            print(e.stderr)


def load_hpc_config(config_path="hpc/configs/tinymistral_config.yaml"):
    """Load HPC configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Sync files with HPC cluster")
    parser.add_argument(
        "--direction",
        choices=["from-hpc", "to-hpc"],
        required=True,
        help="Sync direction",
    )
    parser.add_argument("--host", required=True, help="HPC hostname")
    parser.add_argument("--remote-path", help="Remote path (default from config)")
    parser.add_argument("--local-path", default="./results", help="Local path")
    parser.add_argument(
        "--config",
        default="hpc/configs/tinymistral_config.yaml",
        help="HPC config file",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be synced"
    )

    args = parser.parse_args()

    # Load config if remote path not specified
    if args.remote_path is None:
        if Path(args.config).exists():
            config = load_hpc_config(args.config)
            args.remote_path = config.get("HPC_RESULTS_DIR", "/scratch/user/results")
        else:
            print(f"Config file not found: {args.config}")
            print("Please specify --remote-path")
            return

    # Ensure local directory exists
    Path(args.local_path).mkdir(parents=True, exist_ok=True)

    if args.direction == "from-hpc":
        sync_from_hpc(args.host, args.remote_path, args.local_path, args.dry_run)
    else:
        sync_to_hpc(args.host, args.local_path, args.remote_path, args.dry_run)


if __name__ == "__main__":
    main()
