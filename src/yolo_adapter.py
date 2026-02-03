from ultralytics import YOLO

def yolo_to_detections(model_path, image_path, conf_thres=0.5, allowed_classes=None):
    """
    Returns detections in format:
    [{"cls": "resistor", "conf": 0.91, "xyxy": (x1,y1,x2,y2)}, ...]
    """
    model = YOLO(model_path)
    results = model(image_path)

    r = results[0]
    names = r.names

    dets = []
    if r.boxes is None or len(r.boxes) == 0:
        return dets

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)

    for box, c, k in zip(xyxy, conf, cls):
        if c < conf_thres:
            continue

        cls_name = names[k]  # e.g. "resistor"

        if allowed_classes is not None and cls_name not in allowed_classes:
            continue

        x1, y1, x2, y2 = box
        dets.append({
            "cls": cls_name,
            "conf": float(c),
            "xyxy": (int(x1), int(y1), int(x2), int(y2))
        })

    return dets
