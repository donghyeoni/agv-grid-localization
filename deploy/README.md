# Jetson Nano deployment

The exported ONNX model runs on Jetson Nano with onnxruntime; no PyTorch is
needed on the device.

## Setup (JetPack 4.6, Python 3.6)

```bash
# onnxruntime-gpu wheel built for Jetson (JetPack 4.6 / Python 3.6)
wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl \
     -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install numpy opencv-python
```

OpenCV usually ships with JetPack; if `import cv2` already works, skip the
`opencv-python` install.

## Copy the model and run

Copy the repository and the exported `best.onnx` to the device, then from the
repository root:

```bash
# single image
python3 localize.py --model best.onnx --image photo.jpg --no-show --json out.json

# live camera (CSI or USB)
python3 deploy/camera_demo.py --model best.onnx --csi
python3 deploy/camera_demo.py --model best.onnx --camera 0
```

## Notes

- The model is exported with **opset 12** (`scripts/export_onnx.py`), the
  highest opset the JetPack 4.x onnxruntime builds support.
- Input size is fixed at export time (default 640). Smaller exports
  (`--img 416`) trade accuracy for frame rate on the Nano.
- The first inference is slow (session warm-up); throughput stabilizes from
  the second frame.
