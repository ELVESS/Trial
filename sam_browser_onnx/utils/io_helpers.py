"""
I/O helper functions for image handling and storage.
"""
import base64
import io
import uuid
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image (with or without data URI prefix)

    Returns:
        PIL Image
    """
    # Remove data URI prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    # Decode
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))

    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string.

    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def generate_image_id() -> str:
    """Generate unique image ID."""
    return str(uuid.uuid4())


def save_embedding(embedding: np.ndarray, filepath: Path) -> None:
    """
    Save embedding to .npy file.

    Args:
        embedding: Numpy array
        filepath: Destination path
    """
    np.save(filepath, embedding)


def load_embedding(filepath: Path) -> np.ndarray:
    """
    Load embedding from .npy file.

    Args:
        filepath: Source path

    Returns:
        Numpy array
    """
    return np.load(filepath)


def resize_image_if_needed(
    image: Image.Image,
    max_dimension: int = 1024
) -> Tuple[Image.Image, bool]:
    """
    Resize image if any dimension exceeds max_dimension.
    Maintains aspect ratio.

    Args:
        image: PIL Image
        max_dimension: Maximum width or height

    Returns:
        Tuple of (resized_image, was_resized)
    """
    width, height = image.size

    if width <= max_dimension and height <= max_dimension:
        return image, False

    # Calculate new dimensions
    if width > height:
        new_width = max_dimension
        new_height = int(height * max_dimension / width)
    else:
        new_height = max_dimension
        new_width = int(width * max_dimension / height)

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized, True
