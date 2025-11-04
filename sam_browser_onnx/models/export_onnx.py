#!/usr/bin/env python3
"""
Export SAM mask decoder to ONNX format for browser inference.
Follows the official SAM ONNX export approach.
"""
import os
import sys
import warnings
from pathlib import Path

import torch
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

warnings.filterwarnings("ignore")


def export_decoder_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    model_type: str,
    opset_version: int = 17,
    return_single_mask: bool = True
) -> None:
    """
    Export SAM decoder to ONNX format.

    Args:
        checkpoint_path: Path to .pth checkpoint
        output_path: Path to save .onnx file
        model_type: One of 'vit_b', 'vit_l', 'vit_h'
        opset_version: ONNX opset version
        return_single_mask: If True, return only the best mask (recommended for browser)
    """
    print(f"\nExporting {model_type} decoder to ONNX...")

    # Load SAM model
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))

    # Wrap model for ONNX export
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=False,
        return_extra_metrics=False
    )

    # Determine embedding dimensions based on model type
    if model_type == "vit_b":
        embed_dim = 256
        image_embedding_size = (1, embed_dim, 64, 64)
    elif model_type == "vit_l":
        embed_dim = 256
        image_embedding_size = (1, embed_dim, 64, 64)
    elif model_type == "vit_h":
        embed_dim = 256
        image_embedding_size = (1, embed_dim, 64, 64)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Create dummy inputs for export
    # These shapes define the input spec for the ONNX model
    image_embeddings = torch.randn(image_embedding_size, dtype=torch.float32)
    point_coords = torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float32)
    point_labels = torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32)
    mask_input = torch.randn(1, 1, 256, 256, dtype=torch.float32)
    has_mask_input = torch.tensor([1], dtype=torch.float32)
    orig_im_size = torch.tensor([1024, 1024], dtype=torch.float32)

    # Dynamic axes for variable-length prompts
    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    # Export to ONNX
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        torch.onnx.export(
            onnx_model,
            (
                image_embeddings,
                point_coords,
                point_labels,
                mask_input,
                has_mask_input,
                orig_im_size,
            ),
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=[
                "image_embeddings",
                "point_coords",
                "point_labels",
                "mask_input",
                "has_mask_input",
                "orig_im_size",
            ],
            output_names=["masks", "iou_predictions", "low_res_masks"],
            dynamic_axes=dynamic_axes,
        )

    print(f"✓ Exported to {output_path}")
    print(f"  Model size: {output_path.stat().st_size / (1024*1024):.1f} MB")


def main():
    script_dir = Path(__file__).parent
    checkpoints_dir = script_dir / "checkpoints"
    onnx_dir = script_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SAM ONNX Decoder Exporter")
    print("=" * 60)
    print(f"Checkpoints: {checkpoints_dir.absolute()}")
    print(f"Output: {onnx_dir.absolute()}")

    model_types = ["vit_b", "vit_l", "vit_h"]

    # Check if checkpoints exist
    missing_checkpoints = []
    for model_type in model_types:
        checkpoint_path = checkpoints_dir / f"sam_{model_type}.pth"
        if not checkpoint_path.exists():
            missing_checkpoints.append(checkpoint_path)

    if missing_checkpoints:
        print("\n✗ Missing checkpoints:")
        for cp in missing_checkpoints:
            print(f"  - {cp}")
        print("\nPlease run: python models/download_models.py")
        sys.exit(1)

    # Export each model
    for model_type in model_types:
        checkpoint_path = checkpoints_dir / f"sam_{model_type}.pth"
        output_path = onnx_dir / f"sam_decoder_{model_type}.onnx"

        try:
            export_decoder_to_onnx(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                model_type=model_type,
            )
        except Exception as e:
            print(f"✗ Failed to export {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("✓ ONNX export complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python app.py")
    print("  2. Open: http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
