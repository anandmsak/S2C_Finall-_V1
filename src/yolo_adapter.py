# src/yolo_adapter.py
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "ultralytics is not installed or failed to import. Install with: pip install ultralytics"
    ) from e


# ------------------ NMS (PURE NUMPY) ------------------

def _iou_xyxy(a, b):
    """IoU between two boxes in xyxy format."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    denom = (area_a + area_b - inter)
    if denom <= 0:
        return 0.0
    return inter / denom


def nms_xyxy(boxes, scores, iou_thres=0.45):
    """
    NMS that keeps highest-score boxes and removes overlaps.
    boxes: (N,4) float/int
    scores: (N,) float
    returns list of kept indices
    """
    if len(boxes) == 0:
        return []

    idxs = np.argsort(scores)[::-1]  # high -> low
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        rest = idxs[1:]
        if len(rest) == 0:
            break

        filtered = []
        for j in rest:
            if _iou_xyxy(boxes[i], boxes[j]) <= iou_thres:
                filtered.append(j)

        idxs = np.array(filtered, dtype=np.int64)

    return keep


# ------------------ MAIN ADAPTER ------------------

def yolo_to_detections(
    model_path: str,
    image_path: str,
    conf_thres: float = 0.4,
    iou_thres: float = 0.45,
    allowed_classes=None,
):
    """
    Runs YOLO on image and returns standardized detections:
    [
      {"cls": <name>, "conf": <float>, "xyxy": (x1,y1,x2,y2)},
      ...
    ]

    Fixes:
      ✅ NMS removes duplicate overlapping boxes
      ✅ boxes clipped to image bounds
      ✅ allowed_classes filter (set of class names)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    h, w = img.shape[:2]

    model = YOLO(model_path)
    res = model(image_path, verbose=False)[0]

    if res.boxes is None or len(res.boxes) == 0:
        return []

    # Extract to numpy
    boxes = res.boxes.xyxy.cpu().numpy()   # (N,4)
    scores = res.boxes.conf.cpu().numpy()  # (N,)
    clses = res.boxes.cls.cpu().numpy()    # (N,)

    # Confidence pre-filter (speeds up NMS)
    m = scores >= float(conf_thres)
    boxes = boxes[m]
    scores = scores[m]
    clses = clses[m]

    if len(boxes) == 0:
        return []

    # NMS
    keep = nms_xyxy(boxes, scores, iou_thres=float(iou_thres))

    dets = []
    names = res.names  # dict: id->name

    for i in keep:
        cls_name = names[int(clses[i])]
        if allowed_classes is not None and cls_name not in allowed_classes:
            continue

        x1, y1, x2, y2 = boxes[i].tolist()

        # Clip to image bounds and cast to int
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))

        # Ensure proper ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        dets.append(
            {
                "cls": cls_name,
                "conf": float(scores[i]),
                "xyxy": (x1, y1, x2, y2),
            }
        )

    return dets


# Quick test
if __name__ == "__main__":
    MODEL_PATH = "models/best.pt"
    IMAGE_PATH = "data/raw/input.png"
    allowed = {"resistor", "battery", "capacitor", "transistor"}

    dets = yolo_to_detections(
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        conf_thres=0.35,
        iou_thres=0.45,
        allowed_classes=allowed,
    )

    print("Detections (NMS):")
    for d in dets:
        print(d)
