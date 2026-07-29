"""Configuration constants and defaults for the AGV plate grid detection pipeline."""

import os

# Border cropping thickness (in pixels) applied to each edge of the image.
BORDER_THICKNESS = 30

# Grayscale threshold. Pixels with a value <= this are set to 0 (black).
GRAY_THRESHOLD = 170

# Number of cells per side of the overlaid grid (GRID_SIZE x GRID_SIZE).
GRID_SIZE = 8

# Default demo image shipped with the repository.
DEFAULT_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sample_images",
    "image_0000.jpg",
)
