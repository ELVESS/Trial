"""
Image embedding computation using SAM backbone.
"""
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor


def compute_embedding(predictor: SamPredictor, image: Image.Image) -> np.ndarray:
    """
    Compute image embedding using SAM backbone.

    Args:
        predictor: SamPredictor instance
        image: PIL Image (RGB)

    Returns:
        Image embedding as numpy array (float32)
    """
    # Convert PIL to numpy
    image_np = np.array(image)

    # Set image (this computes the embedding internally)
    predictor.set_image(image_np)

    # Extract the precomputed features
    # The predictor stores them after set_image()
    features = predictor.get_image_embedding().cpu().numpy()

    return features


def get_embedding_shape(model_type: str) -> tuple:
    """
    Get expected embedding shape for a model type.

    Args:
        model_type: One of 'vit_b', 'vit_l', 'vit_h'

    Returns:
        Tuple of (batch, channels, height, width)
    """
    # All SAM models use 256 channels and 64x64 spatial dims
    return (1, 256, 64, 64)
