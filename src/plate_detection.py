"""Plate detection: largest-contour extraction and corner ordering."""

import cv2
import numpy as np


def find_largest_contour(threshold_img):
    """Find the largest external contour in a thresholded image.

    Args:
        threshold_img: Thresholded grayscale image as a ``numpy.ndarray``.

    Returns:
        The largest contour by area, or ``None`` if no contour is found.
    """
    contours, _ = cv2.findContours(
        threshold_img.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def order_corners(contour):
    """Approximate a contour to 4 corners and order them consistently.

    The polygon is approximated with ``approxPolyDP``. The four corners are then
    ordered as (top-left, top-right, bottom-right, bottom-left) using the
    coordinate sum and difference heuristic:

        - top-left     has the smallest (x + y)
        - bottom-right has the largest  (x + y)
        - top-right    has the smallest (x - y) via ``np.diff``
        - bottom-left  has the largest  (x - y) via ``np.diff``

    Args:
        contour: An OpenCV contour.

    Returns:
        A ``(4, 2)`` float32 array of ordered corners, or ``None`` if the
        approximated polygon does not have exactly 4 vertices.
    """
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) != 4:
        return None

    pts = approx.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect
