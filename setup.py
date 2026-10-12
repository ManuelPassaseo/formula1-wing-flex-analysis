"""
Project setup script for the F1 Wing Deflection pipeline.

This script:
  1. Ensures required directories exist.
  2. Downloads the EfficientTAM model checkpoint using the provided shell script.
  3. Downloads the model configuration file (efficienttam_s.yaml) if missing.

"""

import os
import sys
import urllib.request
import shutil


# Configuration
CHECKPOINTS_DIR = "checkpoints"
CONFIGS_DIR = "configs/efficienttam"
CONFIG_FILENAME = "efficienttam_s.yaml"

# URLs
EFFICIENTTAM_CHECKPOINT_URL = (
    "https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s.pt"
)
EFFICIENTTAM_CONFIG_URL = (
    "https://raw.githubusercontent.com/yformer/EfficientTAM/refs/heads/main/efficient_track_anything/configs/efficienttam/efficienttam_s.yaml"
)

# Helper functions
def download_file(url: str, dest_path: str):
    """Download a file from a URL to a local path."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        print(f"Already exists: {dest_path}")
        return

    print(f"Downloading {os.path.basename(dest_path)}...")
    try:
        with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"Downloaded to {dest_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        sys.exit(1)


def main():
    print("Setting up F1 Wing Deflection project...\n")

    # Step 1 — ensure directories exist
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    # Step 2 — download model checkpoint
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, "efficienttam_s.pt")
    download_file(EFFICIENTTAM_CHECKPOINT_URL, checkpoint_path)

    # Step 3 — download config file
    config_path = os.path.join(CONFIGS_DIR, CONFIG_FILENAME)
    download_file(EFFICIENTTAM_CONFIG_URL, config_path)

    print("\nAll assets are ready!")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Config:     {config_path}")
    print("\nSetup complete! You can now run your main script:")
    print("   python main.py\n")


if __name__ == "__main__":
    main()