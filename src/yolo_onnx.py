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


def letterbox(img, new_size=640, color=114, bgr=False):
    """Resize with unchanged aspect ratio and pad to a square.

    Returns:
        ``(padded, scale, (pad_x, pad_y))`` — everything needed to map
        detections back to original pixel coordinates.
    """
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh))
    if bgr:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pad_x, pad_y = (new_size - nw) // 2, (new_size - nh) // 2
    padded = cv2.copyMakeBorder(
        resized, pad_y, new_size - nh - pad_y, pad_x, new_size - nw - pad_x,
        cv2.BORDER_CONSTANT, value=(color, color, color),
    )
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

    def detect(self, img, bgr=False):
        """Detect objects in an RGB image (or a BGR one with ``bgr=True``).

        Returns:
            A list of ``(x1, y1, x2, y2, confidence, class_id)`` tuples in
            original image pixel coordinates.
        """
        padded, scale, (pad_x, pad_y) = letterbox(img, self.input_size, bgr=bgr)
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        blob /= 255.0
        pred = self.session.run(None, {self.input_name: blob})[0][0]

        # pred rows: cx, cy, w, h, objectness, per-class scores
        pred = pred[pred[:, 4] >= self.conf_threshold]
        if not len(pred):
            return []
        scores = pred[:, 4:5] * pred[:, 5:]
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        mask = confidences >= self.conf_threshold
        if not mask.any():
            return []

        pred, confidences, class_ids = pred[mask], confidences[mask], class_ids[mask]
        center, half = pred[:, :2], pred[:, 2:4] / 2
        boxes = np.concatenate([center - half, center + half], axis=1)

        keep = np.concatenate([
            idx[nms(boxes[idx], confidences[idx], self.iou_threshold)]
            for idx in (np.flatnonzero(class_ids == c) for c in np.unique(class_ids))
        ])
        keep = keep[np.argsort(-confidences[keep], kind="stable")]
        boxes = boxes[keep].astype(np.float64)
        boxes[:, 0::2] -= pad_x
        boxes[:, 1::2] -= pad_y
        boxes /= scale
        return [
            (x1, y1, x2, y2, conf, cls_id)
            for (x1, y1, x2, y2), conf, cls_id in zip(
                boxes.tolist(), confidences[keep].tolist(), class_ids[keep].tolist()
            )
        ]
