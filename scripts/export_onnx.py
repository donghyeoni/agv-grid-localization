"""Export trained YOLOv5 weights to ONNX for Jetson Nano.

Jetson Nano (JetPack 4.x) ships onnxruntime/TensorRT builds that support up to
ONNX opset 12, so that is the default here.

Usage:
    python scripts/export_onnx.py                          # runs/agv-grid-8cls/weights/best.pt
    python scripts/export_onnx.py --weights path/to/best.pt --img 640
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = ROOT / "third_party" / "yolov5"
DEFAULT_WEIGHTS = ROOT / "runs" / "agv-grid-8cls" / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"weights not found: {weights} — run scripts/train.py first")
    if not YOLOV5_DIR.exists():
        sys.exit("third_party/yolov5 not found — run scripts/train.py first")

    subprocess.run(
        [sys.executable, str(YOLOV5_DIR / "export.py"),
         "--weights", str(weights),
         "--include", "onnx",
         "--img", str(args.img),
         "--opset", str(args.opset),
         "--simplify"],
        check=True,
        cwd=YOLOV5_DIR,
    )
    print(f"\ndone — ONNX next to {weights} ({weights.with_suffix('.onnx').name})")


if __name__ == "__main__":
    main()
