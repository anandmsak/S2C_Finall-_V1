import cv2
import numpy as np


def _neighbor_count(skel01: np.ndarray) -> np.ndarray:
    """Return 8-neighbor count for each pixel in a 0/1 skeleton image."""
    k = np.array([[1,1,1],
                  [1,0,1],
                  [1,1,1]], dtype=np.uint8)
    # cv2.filter2D works fine with uint8
    return cv2.filter2D(skel01, ddepth=-1, kernel=k)


def _cluster_points(points, radius=6):
    """
    Cluster points that lie within 'radius' pixels using connected components on an impulse image.
    Returns list of representative (x,y) as cluster centroids.
    """
    if len(points) == 0:
        return []

    pts = np.array(points, dtype=np.int32)

    # Build a small binary image just big enough
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    W = (maxx - minx + 1) + 2*radius + 2
    H = (maxy - miny + 1) + 2*radius + 2

    img = np.zeros((H, W), dtype=np.uint8)
    # shift points into this canvas
    sx = radius + 1 - minx
    sy = radius + 1 - miny

    for x, y in points:
        img[y + sy, x + sx] = 255

    # Dilate so close points become one blob
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    blob = cv2.dilate(img, k, iterations=1)

    # Connected components on blob
    n, labels = cv2.connectedComponents((blob > 0).astype(np.uint8), connectivity=8)

    reps = []
    for lab in range(1, n):
        ys, xs = np.where(labels == lab)
        # pick centroid in blob coords
        cx = int(xs.mean())
        cy = int(ys.mean())
        # shift back to original image coords
        reps.append((cx - sx, cy - sy))

    return reps


def prune_spurs(skel, iterations=10):
    """
    Remove tiny spurs: iteratively delete endpoints.
    Helps reduce false junctions.
    """
    sk = (skel > 0).astype(np.uint8)
    for _ in range(iterations):
        nc = _neighbor_count(sk)
        endpoints = ((sk == 1) & (nc == 1)).astype(np.uint8)
        if endpoints.sum() == 0:
            break
        sk[endpoints == 1] = 0
    return (sk * 255).astype(np.uint8)


def find_nodes(skel, cluster_radius=8, spur_prune_iters=5):
    """
    Robust node detection:
    - Optional spur pruning
    - Endpoint candidates: neighbor_count == 1
    - Junction candidates: neighbor_count >= 3
    - Cluster nearby candidates into single nodes

    Returns:
      endpoints: list[(x,y)]
      junctions: list[(x,y)]
      skel_clean: skeleton used
    """
    # 0/1 skeleton
    sk = (skel > 0).astype(np.uint8)

    # Optional: prune spurs to reduce noisy branches
    if spur_prune_iters > 0:
        sk = (prune_spurs(sk*255, iterations=spur_prune_iters) > 0).astype(np.uint8)

    nc = _neighbor_count(sk)

    ep_pts = list(map(tuple, np.column_stack(np.where((sk == 1) & (nc == 1)) )[:, ::-1]))
    jn_pts = list(map(tuple, np.column_stack(np.where((sk == 1) & (nc >= 3)) )[:, ::-1]))

    # Cluster to reduce thousands → handful
    endpoints = _cluster_points(ep_pts, radius=cluster_radius)
    junctions  = _cluster_points(jn_pts, radius=cluster_radius)

    return endpoints, junctions, (sk*255).astype(np.uint8)
