"""Image loading and preprocessing steps.

This module replaces the original Colab notebook workflow (which mounted
Google Drive) with plain local file reading via ``cv2.imread``.
"""

import cv2
import numpy as np


def load_image(path):
    """Load an image from a local path and convert BGR -> RGB.

    Args:
        path: Path to the image file.

    Returns:
        The image as an RGB ``numpy.ndarray``.

    Raises:
        FileNotFoundError: If the image cannot be read from ``path``.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_borders(img, thickness):
    """Blacken a border of the given thickness on all four edges.

    Args:
        img: RGB image as a ``numpy.ndarray``.
        thickness: Border thickness in pixels to set to zero on each edge.

    Returns:
        A copy of ``img`` with the borders masked to black.
    """
    processed = img.copy()
    processed[:thickness, :, :] = 0
    processed[-thickness:, :, :] = 0
    processed[:, :thickness, :] = 0
    processed[:, -thickness:, :] = 0
    return processed


def to_threshold(img, thresh):
    """Convert to grayscale and zero out pixels at or below ``thresh``.

    Pixels with a grayscale value greater than ``thresh`` keep their original
    value; all others are set to 0. This isolates the bright rectangular plate.

    Args:
        img: RGB image as a ``numpy.ndarray``.
        thresh: Grayscale threshold value.

    Returns:
        The thresholded grayscale image as a ``numpy.ndarray``.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.where(gray <= thresh, 0, gray)
