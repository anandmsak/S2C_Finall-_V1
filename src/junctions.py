# src/junctions.py
import cv2
import numpy as np


# -------------------- BASIC UTILS --------------------

def _to01(img: np.ndarray) -> np.ndarray:
    return (img > 0).astype(np.uint8)


def _neighbor_count(skel01: np.ndarray) -> np.ndarray:
    """8-neighbor count for each pixel in a 0/1 skeleton image."""
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel01, ddepth=-1, kernel=k, borderType=cv2.BORDER_CONSTANT)


def _neighbors8(x: int, y: int, w: int, h: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            xx, yy = x + dx, y + dy
            if 0 <= xx < w and 0 <= yy < h:
                yield xx, yy


def _cluster_points(points, radius=6):
    """
    Cluster points within radius using dilation + connected components.
    Returns representative (x,y) centroids.
    """
    if len(points) == 0:
        return []

    pts = np.array(points, dtype=np.int32)
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)

    W = (maxx - minx + 1) + 2 * radius + 2
    H = (maxy - miny + 1) + 2 * radius + 2

    img = np.zeros((H, W), dtype=np.uint8)
    sx = radius + 1 - minx
    sy = radius + 1 - miny

    for x, y in points:
        img[y + sy, x + sx] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    blob = cv2.dilate(img, k, iterations=1)

    n, labels = cv2.connectedComponents((blob > 0).astype(np.uint8), connectivity=8)

    reps = []
    for lab in range(1, n):
        ys, xs = np.where(labels == lab)
        cx = int(xs.mean())
        cy = int(ys.mean())
        reps.append((cx - sx, cy - sy))

    return reps


# -------------------- SPUR REMOVAL (IMPORTANT) --------------------

def prune_short_spurs(skel, max_len=25):
    """
    Remove ONLY short spurs (tiny branches) without shrinking main wires.

    Method:
      - Find endpoints.
      - From each endpoint, walk forward until:
          * reaches a junction/branch (deg != 2), OR
          * exceeds max_len, OR
          * dead-end.
      - If it reaches a junction within <= max_len, remove that walked path.

    max_len:
      15–40 typical. Increase if you still see hair-like branches near junctions.
    """
    sk = _to01(skel)
    h, w = sk.shape[:2]

    nc = _neighbor_count(sk)
    endpoints = list(map(tuple, np.column_stack(np.where((sk == 1) & (nc == 1)))[:, ::-1]))

    # We'll mark pixels to delete
    to_del = set()

    for ex, ey in endpoints:
        path = []
        x, y = ex, ey

        # If pixel is already removed by previous spur deletion, skip
        if sk[y, x] == 0:
            continue

        prev = None
        for step in range(max_len + 1):
            path.append((x, y))

            # recompute degree locally using current skeleton
            deg = 0
            nbrs = []
            for xx, yy in _neighbors8(x, y, w, h):
                if sk[yy, xx] == 1:
                    deg += 1
                    nbrs.append((xx, yy))

            # endpoint start is deg=1, a spur ends when hits junction/branch (deg>=3) or an isolated break (deg==0)
            if (step > 0) and (deg != 2):
                # If we hit a junction/branch quickly, this is a spur -> delete path
                # If deg>=3 => reached a junction
                # If deg==0 or deg==1 => break/endpoint again (still tiny junk)
                for p in path:
                    to_del.add(p)
                break

            if step == max_len:
                # too long => not a spur, keep it
                break

            # choose next pixel (the neighbor that's not prev)
            nxt = None
            for nb in nbrs:
                if prev is None or nb != prev:
                    nxt = nb
                    # For deg==2, there are 2 neighbors; we must avoid going back to prev
                    if prev is not None and nxt == prev and len(nbrs) > 1:
                        continue
                    break

            if nxt is None:
                # dead end
                for p in path:
                    to_del.add(p)
                break

            prev = (x, y)
            x, y = nxt

    # apply deletions
    for (x, y) in to_del:
        sk[y, x] = 0

    return (sk * 255).astype(np.uint8)


# -------------------- NODE FINDING --------------------

def find_nodes(
    skel,
    cluster_radius=10,
    spur_max_len=25,
    junction_deg_min=3,
    suppress_nearby=True
):
    """
    Better node detection:
      1) prune short spurs (not shrinking real wires)
      2) compute neighbor_count
      3) endpoints: deg==1
      4) junctions: deg>=junction_deg_min  (default 3, set 4 if your sketch is super noisy)
      5) cluster nearby candidates

    suppress_nearby:
      removes junctions that are too close to endpoints (common jitter artifact).
    """
    sk = _to01(skel)

    # prune ONLY tiny branches
    if spur_max_len and spur_max_len > 0:
        sk = _to01(prune_short_spurs(sk * 255, max_len=spur_max_len))

    nc = _neighbor_count(sk)

    ep_pts = list(map(tuple, np.column_stack(np.where((sk == 1) & (nc == 1)))[:, ::-1]))
    jn_pts = list(map(tuple, np.column_stack(np.where((sk == 1) & (nc >= junction_deg_min)))[:, ::-1]))

    endpoints = _cluster_points(ep_pts, radius=cluster_radius)
    junctions = _cluster_points(jn_pts, radius=cluster_radius)

    if suppress_nearby and len(junctions) > 0 and len(endpoints) > 0:
        # remove junctions that are basically the same as endpoints / tiny wiggles
        ej = []
        cr2 = float((cluster_radius * 1.2) ** 2)
        ep = np.array(endpoints, dtype=np.float32)
        for (jx, jy) in junctions:
            d2 = ((ep[:, 0] - jx) ** 2 + (ep[:, 1] - jy) ** 2)
            if np.min(d2) > cr2:
                ej.append((jx, jy))
        junctions = ej

    return endpoints, junctions, (sk * 255).astype(np.uint8)
