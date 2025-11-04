"""
Geometry utilities for mask processing.
Includes contour extraction, dilation, and area calculation.
"""
import numpy as np
import cv2
from typing import List, Tuple


def extract_contours(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Extract contours from binary mask.

    Args:
        mask: Binary mask (0/1 or 0/255), shape (H, W)

    Returns:
        List of contours, where each contour is a list of (x, y) points
    """
    # Ensure mask is uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert to list of lists of tuples
    result = []
    for contour in contours:
        # Squeeze and convert to list of (x, y) tuples
        points = contour.squeeze()
        if points.ndim == 1:
            # Single point contour
            points = points.reshape(1, -1)
        if points.ndim == 2 and points.shape[1] == 2:
            result.append([(int(x), int(y)) for x, y in points])

    return result


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Apply morphological dilation to mask.

    Args:
        mask: Binary mask (0/1 or 0/255), shape (H, W)
        radius: Dilation radius in pixels

    Returns:
        Dilated mask (same dtype as input)
    """
    if radius <= 0:
        return mask

    # Ensure mask is uint8
    original_dtype = mask.dtype
    if mask.dtype != np.uint8:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_uint8 = mask

    # Create circular kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1)
    )

    # Dilate
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

    # Convert back to original dtype
    if original_dtype == bool:
        return dilated > 0
    elif original_dtype == np.uint8:
        return dilated
    else:
        return (dilated > 0).astype(original_dtype)


def calculate_mask_area(mask: np.ndarray) -> int:
    """
    Calculate area of binary mask (number of positive pixels).

    Args:
        mask: Binary mask, shape (H, W)

    Returns:
        Number of pixels in mask
    """
    return int(np.sum(mask > 0))


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Convert binary mask to RLE (run-length encoding).
    Useful for compact mask storage.

    Args:
        mask: Binary mask, shape (H, W)

    Returns:
        Dictionary with 'counts' and 'size' keys
    """
    # Flatten mask
    pixels = mask.flatten()

    # Find run lengths
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return {
        'counts': runs.tolist(),
        'size': list(mask.shape)
    }


def rle_to_mask(rle: dict) -> np.ndarray:
    """
    Convert RLE to binary mask.

    Args:
        rle: Dictionary with 'counts' and 'size' keys

    Returns:
        Binary mask
    """
    h, w = rle['size']
    counts = rle['counts']

    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:  # Odd indices are mask pixels
            mask[pos:pos + count] = 1
        pos += count

    return mask.reshape((h, w))
