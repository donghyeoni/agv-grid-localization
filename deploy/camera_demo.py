"""Live camera demo for Jetson Nano: detect objects and print their grid cells.

Grabs frames from a CSI or USB camera, runs the plate-grid + YOLOv5-ONNX +
cell-mapping pipeline on each frame, and prints per-frame results. Run from
the repository root so ``src`` imports resolve:

    python deploy/camera_demo.py --model best.onnx            # USB camera 0
    python deploy/camera_demo.py --model best.onnx --csi      # CSI camera
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import mask_borders, to_threshold
from src.plate_detection import find_largest_contour, order_corners
from src.grid_mapping import boxes_to_cells
from src.yolo_onnx import YoloOnnx, CLASS_NAMES

BORDER_THICKNESS = 30
GRAY_THRESHOLD = 170
GRID_SIZE = 8

CSI_PIPELINE = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
    "framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--csi", action="store_true", help="Use the CSI camera pipeline.")
    parser.add_argument("--camera", type=int, default=0, help="USB camera index.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    args = parser.parse_args()

    model = YoloOnnx(args.model, conf_threshold=args.conf)
    cap = (cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER) if args.csi
           else cv2.VideoCapture(args.camera))
    if not cap.isOpened():
        sys.exit("camera open failed")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.time()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            masked = mask_borders(img_rgb, BORDER_THICKNESS)
            contour = find_largest_contour(to_threshold(masked, GRAY_THRESHOLD))
            corners = order_corners(contour) if contour is not None else None

            detections = model.detect(img_rgb)
            if corners is not None:
                cells = boxes_to_cells([d[:4] for d in detections], corners, args.grid_size)
            else:
                cells = [None] * len(detections)

            elapsed = (time.time() - start) * 1000
            summary = ", ".join(
                f"{CLASS_NAMES[d[5]]}@{c[2] if c else '?'}"
                for d, c in zip(detections, cells)
            ) or "nothing"
            print(f"[{elapsed:6.1f} ms] {summary}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
