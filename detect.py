"""CLI entrypoint for the AGV plate grid detection pipeline.

Pipeline: load -> mask borders -> threshold -> largest contour -> order corners
-> draw grid -> visualize.

Usage:
    python detect.py --image photo.jpg
    python detect.py --image photo.jpg --save out.png --no-show
"""

import argparse
import sys

import cv2

import config
from src.preprocessing import load_image, mask_borders, to_threshold
from src.plate_detection import find_largest_contour, order_corners
from src.grid import draw_grid_on_plate
from src.visualize import show_stages


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a rectangular plate and overlay a perspective-correct grid."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--border-thickness",
        type=int,
        default=config.BORDER_THICKNESS,
        help=f"Border thickness to mask, in pixels (default: {config.BORDER_THICKNESS}).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=config.GRAY_THRESHOLD,
        help=f"Grayscale threshold value (default: {config.GRAY_THRESHOLD}).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=config.GRID_SIZE,
        help=f"Number of grid cells per side (default: {config.GRID_SIZE}).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the result figure (e.g. out.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a matplotlib window (useful for headless runs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and preprocess.
    img_rgb = load_image(args.image)
    img_masked = mask_borders(img_rgb, args.border_thickness)
    threshold_img = to_threshold(img_masked, args.threshold)

    # Detect the plate.
    contour = find_largest_contour(threshold_img)
    if contour is None:
        print(
            "No contour found. Try adjusting --threshold or --border-thickness.",
            file=sys.stderr,
        )
        return 1

    ordered_corners = order_corners(contour)
    if ordered_corners is None:
        print(
            "The largest contour does not approximate to 4 corners; "
            "cannot draw a grid.",
            file=sys.stderr,
        )
        return 1

    # Draw contour and grid overlays on the original color image.
    img_contour = img_rgb.copy()
    cv2.drawContours(img_contour, [contour], -1, (0, 255, 0), 2)

    grid_img = draw_grid_on_plate(img_contour, ordered_corners, args.grid_size)

    # Visualize the pipeline stages.
    show_stages(
        images=[img_rgb, threshold_img, grid_img],
        titles=["Original Image", "Threshold Image (Gray)", "Grid on Plate"],
        save_path=args.save,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
