"""Regenerate the committed artifacts under ``results/`` in one command.

Runs the full plate-grid-detection pipeline on the bundled sample image and
writes both a visual figure and the numeric detection result, so the outcome
stays visible in the repository without re-running.

Artifacts
---------
* ``results/detection.png``  -- original / threshold / grid-overlay stages
* ``results/detection.json`` -- parameters, image shape, contour area and the
  four ordered plate corners (top-left, top-right, bottom-right, bottom-left)

Usage
-----
    python run_all.py
"""

import json
import os

import cv2

import config
from src.preprocessing import load_image, mask_borders, to_threshold
from src.plate_detection import find_largest_contour, order_corners
from src.grid import draw_grid_on_plate
from src.visualize import show_stages

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, "results")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    image_path = config.DEFAULT_IMAGE_PATH
    img_rgb = load_image(image_path)
    img_masked = mask_borders(img_rgb, config.BORDER_THICKNESS)
    threshold_img = to_threshold(img_masked, config.GRAY_THRESHOLD)

    contour = find_largest_contour(threshold_img)
    if contour is None:
        raise SystemExit("No contour found on the bundled sample image.")
    ordered_corners = order_corners(contour)
    if ordered_corners is None:
        raise SystemExit("Largest contour did not approximate to 4 corners.")

    img_contour = img_rgb.copy()
    cv2.drawContours(img_contour, [contour], -1, (0, 255, 0), 2)
    grid_img = draw_grid_on_plate(img_contour, ordered_corners, config.GRID_SIZE)

    show_stages(
        images=[img_rgb, threshold_img, grid_img],
        titles=["Original Image", "Threshold Image (Gray)", "Grid on Plate"],
        save_path=os.path.join(OUT_DIR, "detection.png"),
        show=False,
    )

    result = {
        "image": os.path.relpath(image_path, REPO_ROOT).replace(os.sep, "/"),
        "image_shape": list(img_rgb.shape),
        "params": {
            "border_thickness": config.BORDER_THICKNESS,
            "gray_threshold": config.GRAY_THRESHOLD,
            "grid_size": config.GRID_SIZE,
        },
        "contour_area_px": float(cv2.contourArea(contour)),
        "ordered_corners_tl_tr_br_bl": ordered_corners.tolist(),
    }
    with open(os.path.join(OUT_DIR, "detection.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("Detected corners (TL, TR, BR, BL):")
    print(ordered_corners)
    print(f"Contour area: {result['contour_area_px']:.1f} px")
    print(f"Artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
