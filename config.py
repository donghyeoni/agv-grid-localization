"""Configuration constants and defaults for the AGV plate grid detection pipeline."""

# Border cropping thickness (in pixels) applied to each edge of the image.
BORDER_THICKNESS = 30

# Grayscale threshold. Pixels with a value <= this are set to 0 (black).
GRAY_THRESHOLD = 170

# Number of cells per side of the overlaid grid (GRID_SIZE x GRID_SIZE).
GRID_SIZE = 8
