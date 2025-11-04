#!/usr/bin/env python3
"""
Download ONNX Runtime Web library.
This is a helper script to download the required JavaScript library.
"""
import urllib.request
import sys
from pathlib import Path

def download_onnx_web():
    """Download ONNX Runtime Web library."""
    # Try multiple CDN sources
    urls = [
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js",
        "https://unpkg.com/onnxruntime-web@1.16.3/dist/ort.min.js",
        "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-1.16.3.tgz",
    ]

    dest_dir = Path(__file__).parent / "static" / "vendor"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "onnxruntime-web.min.js"

    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"✓ {dest_path.name} already exists ({dest_path.stat().st_size} bytes)")
        return

    print("Downloading ONNX Runtime Web library...")
    print("This is required for browser-side inference.")

    for url in urls[:2]:  # Try the JS URLs first
        try:
            print(f"\nTrying: {url}")
            urllib.request.urlretrieve(url, dest_path)

            # Check if download was successful
            if dest_path.stat().st_size > 1000:
                print(f"✓ Downloaded successfully ({dest_path.stat().st_size} bytes)")
                return
            else:
                print(f"✗ Download failed (file too small)")
                dest_path.unlink()
        except Exception as e:
            print(f"✗ Failed: {e}")
            if dest_path.exists():
                dest_path.unlink()

    # If all automatic downloads fail, provide manual instructions
    print("\n" + "="*60)
    print("⚠️  Automatic download failed")
    print("="*60)
    print("\nPlease download manually:")
    print("\n1. Visit this URL in your browser:")
    print("   https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js")
    print("\n2. Save the file to:")
    print(f"   {dest_path.absolute()}")
    print("\n3. Or use curl/wget manually:")
    print(f"   curl -L {urls[0]} -o {dest_path}")
    print("="*60)
    sys.exit(1)

if __name__ == "__main__":
    download_onnx_web()
