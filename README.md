# AGV Plate Grid Detection

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

A classical computer-vision (OpenCV) pipeline that localizes a rectangular plate
in a single overhead photo and overlays a perspective-correct 8x8 grid on it.
This is the **localization step** for an AGV (Automated Guided Vehicle): given
one top-down image, it finds the plate and produces a grid that maps the plate
surface into discrete cells.

## Overview

The pipeline is a sequence of deterministic OpenCV steps (no learned model):

1. **Load** the image and convert BGR -> RGB.
2. **Mask borders** by blacking out a fixed-thickness margin on every edge, which
   removes edge clutter that would otherwise confuse contour detection.
3. **Threshold** the grayscale image (pixels at or below the threshold become 0)
   to isolate the bright rectangular plate.
4. **Find the largest external contour**, assumed to be the plate.
5. **Order the 4 corners** via `approxPolyDP` and a coordinate sum/difference
   heuristic (top-left = min x+y, bottom-right = max x+y, top-right = min x-y,
   bottom-left = max x-y).
6. **Draw a perspective-correct grid** by linearly interpolating between opposite
   edges of the quadrilateral, so the grid follows the plate's perspective.
7. **Visualize** the original, threshold, and grid-overlay stages side by side.

## Dataset

The original project folder shipped with a Roboflow **"hihi" v2** export in the
**YOLOv5-PyTorch** format (delivered as `.zip` archives), plus a separate
`dataset.zip`. **None of these are used by this notebook or by this repository.**
This repo is a *classical-CV grid-detection step* only — there is **no trained
machine-learning model** here, and no object detection is performed. The YOLO
export is mentioned purely for provenance/honesty.

The dataset archives are **not included** in this repository. The demo instead
runs on a single bundled sample image, `sample_images/image_0000.jpg`.

## Structure

```
agv-plate-grid-detection/
├── config.py                 # Constants: border thickness, threshold, grid size, default image
├── detect.py                 # argparse CLI wiring the full pipeline
├── requirements.txt
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # load_image, mask_borders, to_threshold
│   ├── plate_detection.py    # find_largest_contour, order_corners
│   ├── grid.py               # draw_grid_on_plate
│   └── visualize.py          # show_stages (1xN matplotlib helper)
├── notebooks/
│   └── detection.ipynb       # Original notebook, kept verbatim
└── sample_images/
    └── image_0000.jpg        # Tiny demo input (~17 KB)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run on the bundled sample image (default):

```bash
python detect.py
```

Run on your own image, or tune parameters:

```bash
python detect.py --image path/to/photo.jpg
python detect.py --image sample_images/image_0000.jpg --threshold 170 --grid-size 8
python detect.py --save out.png --no-show      # headless: save figure, no window
```

CLI options:

| Option | Default | Description |
| --- | --- | --- |
| `--image` | bundled sample | Path to the input image |
| `--border-thickness` | `30` | Border margin (px) masked on each edge |
| `--threshold` | `170` | Grayscale threshold value |
| `--grid-size` | `8` | Grid cells per side (`N x N`) |
| `--save` | none | Save the result figure to this path |
| `--no-show` | off | Skip opening a matplotlib window |

## Notes

- **Classical CV, not ML.** The pipeline relies on thresholding and contour
  geometry. It assumes a bright, roughly rectangular plate that is the largest
  bright region after border masking.
- **Parameter sensitivity.** The default threshold (`170`) and border thickness
  (`30`) were tuned for the sample image. Different lighting or camera setups may
  require adjusting `--threshold` and `--border-thickness`.
- **Failure modes.** If no contour is found, or the largest contour does not
  approximate to exactly 4 corners, `detect.py` reports the issue and exits
  without drawing a grid.
- **Provenance.** The code was refactored from the original Colab notebook
  (`notebooks/detection.ipynb`). Google Drive mounting and `!pip` boilerplate
  were removed in favor of local file reading and `requirements.txt`.
