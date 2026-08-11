"""Train YOLOv5 on the merged 8-class dataset.

Clones ultralytics/yolov5 (v7.0) into third_party/ on first run, then invokes
its train.py. Run scripts/prepare_dataset.py first.

Usage:
    python scripts/train.py                     # defaults: yolov5s, 100 epochs
    python scripts/train.py --epochs 150 --batch 16 --img 640
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLOV5_DIR = ROOT / "third_party" / "yolov5"
YOLOV5_REPO = "https://github.com/ultralytics/yolov5"
YOLOV5_TAG = "v7.0"
DATA_YAML = ROOT / "data" / "merged" / "data.yaml"


def ensure_yolov5() -> None:
    if YOLOV5_DIR.exists():
        return
    YOLOV5_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", YOLOV5_TAG,
         YOLOV5_REPO, str(YOLOV5_DIR)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov5s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="agv-grid-8cls")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        sys.exit("data/merged/data.yaml not found — run scripts/prepare_dataset.py first")

    ensure_yolov5()
    subprocess.run(
        [sys.executable, str(YOLOV5_DIR / "train.py"),
         "--data", str(DATA_YAML),
         "--weights", args.weights,
         "--epochs", str(args.epochs),
         "--batch-size", str(args.batch),
         "--img", str(args.img),
         "--device", args.device,
         "--project", str(ROOT / "runs"),
         "--name", args.name,
         "--exist-ok"],
        check=True,
        cwd=YOLOV5_DIR,
    )
    print(f"\ndone — weights at runs/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
