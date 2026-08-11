"""Merge the two Roboflow exports into one 8-class YOLOv5 dataset.

Inputs (under data/, not tracked in git):
  - kkk8-detection.v2i.yolov5pytorch  (8 doll classes)
  - agvdetect.v1i.yolov5pytorch       (1 class: AGV)

Transformations:
  - DOG is renamed to DARK_GRAY (same label id, name change only).
  - RATEL boxes are dropped entirely.
  - AGV is appended as the last class.

Output: data/merged/ with train/valid/test splits and data.yaml.

Usage:
    python scripts/prepare_dataset.py
"""

import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOLLS_DIR = DATA_DIR / "kkk8-detection.v2i.yolov5pytorch"
AGV_DIR = DATA_DIR / "agvdetect.v1i.yolov5pytorch"
OUT_DIR = DATA_DIR / "merged"

CLASSES = [
    "BLACK", "DARK_GRAY", "GRAY_BIG", "GRAY_SMALL",
    "PINK", "SKYBLUE", "WHITE", "AGV",
]

# kkk8 label id -> merged id (None = drop)
DOLL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: None, 6: 5, 7: 6}  # 5 = RATEL
AGV_MAP = {0: 7}

SPLITS = ["train", "valid", "test"]


def convert(src_dir: Path, id_map: dict, prefix: str) -> dict:
    stats = {"images": 0, "boxes": 0, "dropped": 0}
    for split in SPLITS:
        img_out = OUT_DIR / split / "images"
        lbl_out = OUT_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for lbl in sorted((src_dir / split / "labels").glob("*.txt")):
            img = src_dir / split / "images" / (lbl.stem + ".jpg")
            if not img.exists():
                continue
            lines = []
            for raw in lbl.read_text().splitlines():
                parts = raw.split()
                if not parts:
                    continue
                new_id = id_map.get(int(parts[0]))
                if new_id is None:
                    stats["dropped"] += 1
                    continue
                lines.append(" ".join([str(new_id)] + parts[1:]))
            shutil.copy2(img, img_out / f"{prefix}_{img.name}")
            (lbl_out / f"{prefix}_{lbl.name}").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            stats["images"] += 1
            stats["boxes"] += len(lines)
    return stats


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    doll_stats = convert(DOLLS_DIR, DOLL_MAP, "doll")
    agv_stats = convert(AGV_DIR, AGV_MAP, "agv")

    yaml_lines = [
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(CLASSES)}",
        f"names: {CLASSES}",
    ]
    (OUT_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    print(f"dolls: {doll_stats}")
    print(f"agv:   {agv_stats}")
    print(f"classes ({len(CLASSES)}): {CLASSES}")
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()
