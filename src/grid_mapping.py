"""Map detected object boxes to grid cells on the plate.

The plate corners define a perspective transform from image space to a
normalized grid space. An object's bounding-box center is projected through
that transform, and the cell it lands in is reported as (row, col) plus a
flat cell index (row-major, 0 at the top-left).
"""

import cv2
import numpy as np


def grid_transform(ordered_corners, grid_size):
    """Build the image -> grid-space perspective transform.

    Args:
        ordered_corners: A ``(4, 2)`` array of plate corners ordered as
            (top-left, top-right, bottom-right, bottom-left).
        grid_size: Number of cells per side.

    Returns:
        A ``3x3`` perspective matrix mapping image pixels to grid coordinates
        in ``[0, grid_size]`` on both axes.
    """
    src = np.asarray(ordered_corners, dtype="float32")
    dst = np.array(
        [[0, 0], [grid_size, 0], [grid_size, grid_size], [0, grid_size]],
        dtype="float32",
    )
    return cv2.getPerspectiveTransform(src, dst)


def boxes_to_cells(boxes, ordered_corners, grid_size):
    """Assign each detection box to a grid cell by its center point.

    Args:
        boxes: Iterable of ``(x1, y1, x2, y2)`` boxes in pixel coordinates.
        ordered_corners: Plate corners as in :func:`grid_transform`.
        grid_size: Number of cells per side.

    Returns:
        A list the same length as ``boxes``: each element is
        ``(row, col, index)`` with ``index = row * grid_size + col``, or
        ``None`` for boxes centered off the plate.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(boxes):
        return []
    centers = ((boxes[:, :2] + boxes[:, 2:]) / 2.0).astype(np.float32)
    grid = cv2.perspectiveTransform(centers[:, None], grid_transform(ordered_corners, grid_size))[:, 0]
    inside = ((grid >= 0) & (grid < grid_size)).all(axis=1)
    cols, rows = np.where(inside[:, None], grid, 0).astype(int).T
    return [
        (row, col, row * grid_size + col) if ok else None
        for row, col, ok in zip(rows.tolist(), cols.tolist(), inside.tolist())
    ]
