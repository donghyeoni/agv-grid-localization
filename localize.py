"""Full pipeline: plate grid + YOLOv5 (ONNX) detections + grid-cell mapping.

Pipeline: detect the plate and its 8x8 grid (classical CV), run the exported
YOLOv5 ONNX model on the same image, then map each detection's box center to
the grid cell it occupies. Prints a table of (class, confidence, cell) and can
save an annotated figure plus a machine-readable JSON.

Usage:
    python localize.py --model best.onnx --image path/to/photo.jpg --save out.png
"""

import argparse
import json
import sys

import cv2

from src.preprocessing import load_image, mask_borders, to_threshold
from src.plate_detection import find_largest_contour, order_corners
from src.grid import draw_grid_on_plate
from src.grid_mapping import boxes_to_cells
from src.yolo_onnx import YoloOnnx, CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect objects on the plate and report their grid cells."
    )
    parser.add_argument("--model", required=True, help="Path to the YOLOv5 ONNX model.")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument("--border-thickness", type=int, default=30,
                        help="Border margin (px) masked on each edge.")
    parser.add_argument("--threshold", type=int, default=170,
                        help="Grayscale threshold for plate isolation.")
    parser.add_argument("--grid-size", type=int, default=8,
                        help="Grid cells per side (N x N).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--save", default=None, help="Save the annotated image here.")
    parser.add_argument("--json", default=None, help="Save detections as JSON here.")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Stage 1: plate grid (classical CV).
    img_rgb = load_image(args.image)
    img_masked = mask_borders(img_rgb, args.border_thickness)
    threshold_img = to_threshold(img_masked, args.threshold)
    contour = find_largest_contour(threshold_img)
    if contour is None:
        print("No plate contour found.", file=sys.stderr)
        return 1
    corners = order_corners(contour)
    if corners is None:
        print("Plate contour does not approximate to 4 corners.", file=sys.stderr)
        return 1

    # Stage 2: object detection (YOLOv5 ONNX).
    model = YoloOnnx(args.model, conf_threshold=args.conf)
    detections = model.detect(img_rgb)

    # Stage 3: map each detection to its grid cell.
    boxes = [d[:4] for d in detections]
    cells = boxes_to_cells(boxes, corners, args.grid_size)

    annotated = draw_grid_on_plate(img_rgb, corners, args.grid_size)
    records = []
    print(f"{'class':<12} {'conf':>5}  cell")
    for (x1, y1, x2, y2, conf, cls_id), cell in zip(detections, cells):
        name = CLASS_NAMES[cls_id]
        cell_str = f"({cell[0]},{cell[1]}) #{cell[2]}" if cell else "off-plate"
        print(f"{name:<12} {conf:>5.2f}  {cell_str}")
        records.append(
            {
                "class": name,
                "confidence": round(conf, 4),
                "box": [round(v, 1) for v in (x1, y1, x2, y2)],
                "cell": None if cell is None else
                        {"row": cell[0], "col": cell[1], "index": cell[2]},
            }
        )
        label = f"{name} #{cell[2]}" if cell else name
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.putText(annotated, label, (int(x1), int(y1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"image": args.image, "detections": records}, f, indent=2)
    if args.save:
        cv2.imwrite(args.save, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.imshow(annotated)
        plt.axis("off")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
