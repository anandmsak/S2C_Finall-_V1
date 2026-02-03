import cv2
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)


# ---------------- COMPONENT MASKING ----------------

def detections_to_mask(img_shape, detections, pad=18):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if detections is None:
        return mask

    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def mask_components(frame, detections, pad=18):
    if detections is None:
        return frame

    h, w = frame.shape[:2]
    masked = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
        cv2.rectangle(masked, (x1, y1), (x2, y2), (255, 255, 255), -1)

    return masked


# ---------------- STRONGER WIRE EXTRACTION ----------------

def wire_binary(masked_bgr):
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 7
    )

    # REMOVE small noise specks
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_small, iterations=1)

    # CLOSE small gaps in wires
    k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_line, iterations=1)

    k_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_line_v, iterations=1)

    return bw


# ---------------- SKELETON ----------------

def skeletonize(binary):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary)

    img = binary.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skel


# ---------------- MAIN PIPELINE ----------------

def preprocess_wires(image_path, detections=None, pad=18):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    masked = mask_components(img, detections, pad)
    wires_bin = wire_binary(masked)

    comp_mask = detections_to_mask(img.shape, detections, pad)
    wires_bin = cv2.bitwise_and(wires_bin, cv2.bitwise_not(comp_mask))

    wires_skel = skeletonize(wires_bin)

    return img, masked, comp_mask, wires_bin, wires_skel


# ---------------- EXECUTION ----------------

def main():
    from src.yolo_adapter import yolo_to_detections
    from src.junctions import find_nodes
    from src.wire_graph import (
        build_graph_from_skeleton,
        draw_graph_debug,
        merge_close_nodes,
        remap_edges,
        compress_degree2
    )

    IMAGE_PATH = "data/raw/input.png"
    MODEL_PATH = "models/best.pt"
    REMOVE_CLASSES = {"resistor", "battery", "capacitor", "transistor"}

    detections = yolo_to_detections(
        MODEL_PATH, IMAGE_PATH,
        conf_thres=0.4,
        iou_thres=0.45,
        allowed_classes=REMOVE_CLASSES
    )

    img, masked, comp_mask, wires_bin, wires_skel = preprocess_wires(
        IMAGE_PATH, detections, pad=18
    )

    cv2.imwrite("data/processed/wires_only.png", wires_bin)
    cv2.imwrite("data/processed/wires_skeleton.png", wires_skel)

    # NODE DETECTION (LESS SENSITIVE)
    endpoints, junctions, skel_clean = find_nodes(
        wires_skel,
        cluster_radius=12,
        spur_max_len=25,
        junction_deg_min=3
    )

    # BUILD GRAPH (SNAP FIX)
    nodes, edges, _ = build_graph_from_skeleton(
        skel_clean, endpoints, junctions,
        node_radius=12,
        min_seg_area=40,
        endpoint_snap=22
    )

    # MERGE JITTER NODES
    nodes_m, mapping = merge_close_nodes(nodes, merge_radius=40)
    edges_m = remap_edges(edges, mapping)

    # SIMPLIFY STRAIGHT WIRES
    nodes2, edges2 = compress_degree2(nodes_m, edges_m)

    print("Final Nodes:", len(nodes2))
    print("Final Edges:", len(edges2))

    dbg = draw_graph_debug(skel_clean, nodes2, edges2)
    cv2.imwrite("data/processed/graph_final.png", dbg)


if __name__ == "__main__":
    main()
