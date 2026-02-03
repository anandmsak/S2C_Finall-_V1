import cv2
import numpy as np
import os

# Always ensure output folder exists
os.makedirs("data/processed", exist_ok=True)


def detections_to_mask(img_shape, detections, pad=18):
    """Create a binary mask (255 inside detected component boxes, 0 elsewhere)."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if detections is None:
        return mask

    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    return mask


def mask_components(frame, detections, pad=18):
    """Paint detected component regions white in the original BGR image."""
    if detections is None:
        return frame

    h, w = frame.shape[:2]
    masked = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
        cv2.rectangle(masked, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

    return masked


def wire_binary(masked_bgr):
    """Convert image to binary where wires are white (255) and background black (0)."""
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=8
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    return bw


def skeletonize(binary):
    """Skeletonize binary (wires=255)."""
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


def preprocess_wires(image_path, detections=None, pad=18, debug=False):
    """
    Pipeline:
      1) Mask components in BGR
      2) Threshold to binary wires
      3) HARD remove anything inside detection boxes from binary
      4) Close tiny gaps
      5) Skeletonize
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    masked = mask_components(img, detections, pad=pad)
    wires_bin = wire_binary(masked)

    comp_mask = detections_to_mask(img.shape, detections, pad=pad)
    wires_bin_clean = cv2.bitwise_and(wires_bin, cv2.bitwise_not(comp_mask))

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wires_bin_clean = cv2.morphologyEx(wires_bin_clean, cv2.MORPH_CLOSE, k, iterations=1)

    wires_skel = skeletonize(wires_bin_clean)

    if debug:
        cv2.imshow("Original", img)
        cv2.imshow("Masked (components removed)", masked)
        cv2.imshow("Component Mask (binary)", comp_mask)
        cv2.imshow("Wires Binary (clean)", wires_bin_clean)
        cv2.imshow("Wires Skeleton", wires_skel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img, masked, comp_mask, wires_bin_clean, wires_skel


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

    # 1) YOLO detections -> standardized format
    detections = yolo_to_detections(
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        conf_thres=0.5,   # try 0.3 / 0.25 if capacitor not detected
        allowed_classes=REMOVE_CLASSES
    )
    print("detections:", detections)

    # 2) Wire preprocessing
    img, masked, comp_mask, wires_bin, wires_skel = preprocess_wires(
        IMAGE_PATH, detections=detections, pad=18, debug=False
    )

    cv2.imwrite("data/processed/masked.png", masked)
    cv2.imwrite("data/processed/wires_only.png", wires_bin)
    cv2.imwrite("data/processed/wires_skeleton.png", wires_skel)
    cv2.imwrite("data/processed/component_mask.png", comp_mask)

    # 3) Node detection (cluster + spur prune)
    endpoints, junctions, skel_clean = find_nodes(
        wires_skel, cluster_radius=10, spur_prune_iters=8
    )
    print("Endpoints:", len(endpoints))
    print("Junctions:", len(junctions))

    cv2.imwrite("data/processed/skeleton_clean.png", skel_clean)

    # 4) Build initial graph
    nodes, edges, sk_no_nodes = build_graph_from_skeleton(
    skel_clean, endpoints, junctions,
    node_radius=6,
    min_seg_area=40,
    endpoint_snap=16
    )

    print("Graph nodes:", len(nodes))
    print("Graph edges:", len(edges))

    cv2.imwrite("data/processed/skel_no_nodes.png", sk_no_nodes)

    dbg = draw_graph_debug(skel_clean, nodes, edges)
    cv2.imwrite("data/processed/graph_debug.png", dbg)

    # 5) NEW: Merge close nodes BEFORE degree-2 compression
    # Increase merge_radius if you still see many nodes on straight wires.
    nodes_m, mapping = merge_close_nodes(nodes, merge_radius=30)   
    edges_m = remap_edges(edges, mapping)

    print("Merged nodes:", len(nodes_m))
    print("Merged edges:", len(edges_m))

    dbg_m = draw_graph_debug(skel_clean, nodes_m, edges_m)
    cv2.imwrite("data/processed/graph_debug_merged.png", dbg_m)

    # 6) Simplify graph (compress degree-2 nodes)
    nodes2, edges2 = compress_degree2(nodes_m, edges_m)
    print("Simplified nodes:", len(nodes2))
    print("Simplified edges:", len(edges2))

    dbg2 = draw_graph_debug(skel_clean, nodes2, edges2)
    cv2.imwrite("data/processed/graph_debug_simplified.png", dbg2)

    # 7) Optional display
    cv2.imshow("Graph Debug (raw)", dbg)
    cv2.imshow("Graph Debug (merged)", dbg_m)
    cv2.imshow("Graph Debug (simplified)", dbg2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # run:
    # python -m src.wire_preprocess
    main()
