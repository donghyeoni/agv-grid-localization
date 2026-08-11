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


def point_to_cell(point, matrix, grid_size):
    """Project an image point into the grid and return its cell.

    Args:
        point: ``(x, y)`` pixel coordinates.
        matrix: Transform from :func:`grid_transform`.
        grid_size: Number of cells per side.

    Returns:
        ``(row, col, index)`` with ``index = row * grid_size + col``, or
        ``None`` if the point falls outside the plate.
    """
    src = np.array([[point]], dtype="float32")
    gx, gy = cv2.perspectiveTransform(src, matrix)[0, 0]
    if not (0 <= gx < grid_size and 0 <= gy < grid_size):
        return None
    row, col = int(gy), int(gx)
    return row, col, row * grid_size + col


def boxes_to_cells(boxes, ordered_corners, grid_size):
    """Assign each detection box to a grid cell by its center point.

    Args:
        boxes: Iterable of ``(x1, y1, x2, y2)`` boxes in pixel coordinates.
        ordered_corners: Plate corners as in :func:`grid_transform`.
        grid_size: Number of cells per side.

    Returns:
        A list the same length as ``boxes``: each element is
        ``(row, col, index)`` or ``None`` for boxes centered off the plate.
    """
    matrix = grid_transform(ordered_corners, grid_size)
    cells = []
    for x1, y1, x2, y2 in boxes:
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        cells.append(point_to_cell(center, matrix, grid_size))
    return cells
