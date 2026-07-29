"""Perspective-correct grid overlay drawing."""

import cv2
import numpy as np


def draw_grid_on_plate(img, ordered_corners, grid_size):
    """Draw a perspective-correct grid over the plate defined by 4 corners.

    Grid lines are drawn by linearly interpolating between opposite edges of
    the quadrilateral, so the grid follows the plate's perspective.

    Args:
        img: RGB image to draw on (not modified in place).
        ordered_corners: A ``(4, 2)`` array of corners ordered as
            (top-left, top-right, bottom-right, bottom-left).
        grid_size: Number of cells per side (``grid_size`` x ``grid_size``).

    Returns:
        A copy of ``img`` with the grid drawn in green.
    """
    (tl, tr, br, bl) = ordered_corners
    result_img = img.copy()

    for i in range(grid_size + 1):
        ratio = i / grid_size

        # Horizontal line: interpolate between the left and right edges.
        start_h = tl + (bl - tl) * ratio
        end_h = tr + (br - tr) * ratio
        cv2.line(
            result_img,
            tuple(start_h.astype(int)),
            tuple(end_h.astype(int)),
            (0, 255, 0),
            1,
        )

        # Vertical line: interpolate between the top and bottom edges.
        start_v = tl + (tr - tl) * ratio
        end_v = bl + (br - bl) * ratio
        cv2.line(
            result_img,
            tuple(start_v.astype(int)),
            tuple(end_v.astype(int)),
            (0, 255, 0),
            1,
        )

    return result_img
