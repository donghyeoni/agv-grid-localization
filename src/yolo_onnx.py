"""Minimal ONNX Runtime inference for exported YOLOv5 models.

Self-contained (numpy + cv2 + onnxruntime) so the exact same code runs on the
development machine and on Jetson Nano.
"""

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = [
    "BLACK", "DARK_GRAY", "GRAY_BIG", "GRAY_SMALL",
    "PINK", "SKYBLUE", "WHITE", "AGV",
]


def letterbox(img, new_size=640, color=114):
    """Resize with unchanged aspect ratio and pad to a square.

    Returns:
        ``(padded, scale, (pad_x, pad_y))`` — everything needed to map
        detections back to original pixel coordinates.
    """
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_x, pad_y = (new_size - nw) // 2, (new_size - nh) // 2
    padded = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return padded, scale, (pad_x, pad_y)


def nms(boxes, scores, iou_threshold=0.45):
    """Pure-numpy non-maximum suppression; returns kept indices."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


class YoloOnnx:
    """Run an exported YOLOv5 ONNX model on RGB images."""

    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.session = ort.InferenceSession(
            str(model_path), providers=ort.get_available_providers()
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.session.get_inputs()[0].shape[2]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, img_rgb):
        """Detect objects in an RGB image.

        Returns:
            A list of ``(x1, y1, x2, y2, confidence, class_id)`` tuples in
            original image pixel coordinates.
        """
        padded, scale, (pad_x, pad_y) = letterbox(img_rgb, self.input_size)
        blob = padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        pred = self.session.run(None, {self.input_name: blob})[0][0]

        # pred rows: cx, cy, w, h, objectness, per-class scores
        scores = pred[:, 4:5] * pred[:, 5:]
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        mask = confidences >= self.conf_threshold
        if not mask.any():
            return []

        cxcywh = pred[mask, :4]
        confidences, class_ids = confidences[mask], class_ids[mask]
        boxes = np.empty_like(cxcywh)
        boxes[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        boxes[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        boxes[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        boxes[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

        keep = nms(boxes, confidences, self.iou_threshold)
        detections = []
        for i in keep:
            x1 = (boxes[i, 0] - pad_x) / scale
            y1 = (boxes[i, 1] - pad_y) / scale
            x2 = (boxes[i, 2] - pad_x) / scale
            y2 = (boxes[i, 3] - pad_y) / scale
            detections.append(
                (float(x1), float(y1), float(x2), float(y2),
                 float(confidences[i]), int(class_ids[i]))
            )
        return detections
