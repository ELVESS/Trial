#!/usr/bin/env python3
"""
Download official SAM model checkpoints from Facebook Research.
Checkpoints are saved to ./checkpoints/
"""
import os
import sys
import urllib.request
from pathlib import Path

# Official SAM checkpoint URLs
CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress bar."""
    if dest_path.exists():
        print(f"✓ {dest_path.name} already exists, skipping.")
        return

    print(f"Downloading {dest_path.name} from {url}...")
    print("This may take several minutes depending on your connection.")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}% ({count * block_size // (1024*1024)} MB / {total_size // (1024*1024)} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print(f"\n✓ Downloaded {dest_path.name}")
    except Exception as e:
        print(f"\n✗ Failed to download {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise

def main():
    # Get checkpoints directory
    script_dir = Path(__file__).parent
    checkpoints_dir = script_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SAM Model Checkpoint Downloader")
    print("=" * 60)
    print(f"Destination: {checkpoints_dir.absolute()}\n")

    # Download each checkpoint
    for model_type, url in CHECKPOINT_URLS.items():
        filename = url.split("/")[-1]
        dest_path = checkpoints_dir / f"sam_{model_type}.pth"
        download_file(url, dest_path)

    print("\n" + "=" * 60)
    print("✓ All checkpoints downloaded successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python models/export_onnx.py")
    print("  2. Run: python app.py")

if __name__ == "__main__":
    main()
