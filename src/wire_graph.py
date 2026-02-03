import cv2
import numpy as np


def _to01(img):
    return (img > 0).astype(np.uint8)


# ------------------ NODE REGION LABELING (ROBUST) ------------------

def label_node_regions_voronoi(nodes, shape, radius=8):
    """
    Every pixel inside 'radius' of a node is assigned to the NEAREST node.
    Returns:
      node_lab: int32 label image (node_id+1), 0 elsewhere
      node_mask: uint8 mask (1 where any node region)
    """
    h, w = shape[:2]
    node_lab = np.zeros((h, w), dtype=np.int32)

    if len(nodes) == 0:
        return node_lab, np.zeros((h, w), dtype=np.uint8)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    pts = np.array(nodes, dtype=np.float32)  # (N,2) in (x,y)
    dist2 = (xx[None, :, :] - pts[:, 0][:, None, None])**2 + (yy[None, :, :] - pts[:, 1][:, None, None])**2

    nearest = np.argmin(dist2, axis=0)
    mind2 = np.min(dist2, axis=0)

    r2 = float(radius * radius)
    inside = (mind2 <= r2)

    node_lab[inside] = nearest[inside].astype(np.int32) + 1
    node_mask = inside.astype(np.uint8)

    return node_lab, node_mask


# ------------------ SEGMENT ENDPOINT FINDER ------------------

def segment_endpoints(seg_img_01):
    """
    Given a single connected segment (0/1 image),
    endpoints = pixels with exactly 1 neighbor in 8-connectivity.
    """
    seg = seg_img_01.astype(np.uint8)
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=np.uint8)
    neigh = cv2.filter2D(seg, -1, k, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((seg == 1) & (neigh == 1))
    return list(zip(xs.tolist(), ys.tolist()))


def nearest_node_for_point_with_dist(pt, nodes):
    """
    Return (node_index, dist2) for nearest node, or (None, None) if nodes empty.
    """
    if len(nodes) == 0:
        return None, None
    x, y = pt
    pts = np.array(nodes, dtype=np.float32)
    d2 = (pts[:, 0] - x)**2 + (pts[:, 1] - y)**2
    i = int(np.argmin(d2))
    return i, float(d2[i])


def farthest_two_points(points_xy):
    """
    points_xy: list[(x,y)]
    Returns 2 points that are farthest apart (approx, O(n)).
    """
    if len(points_xy) < 2:
        return None

    pts = np.array(points_xy, dtype=np.float32)
    a = pts[0]
    d2 = np.sum((pts - a) ** 2, axis=1)
    b = pts[np.argmax(d2)]
    d2b = np.sum((pts - b) ** 2, axis=1)
    c = pts[np.argmax(d2b)]
    return (int(b[0]), int(b[1])), (int(c[0]), int(c[1]))


# ------------------ BUILD GRAPH FROM SKELETON ------------------

def build_graph_from_skeleton(
    skel, endpoints, junctions,
    node_radius=8,
    min_seg_area=30,
    endpoint_snap=14,
    max_endpoints_check=40,
    min_edge_pixels=25
):
    """
    Robust skeleton-to-graph (IMPROVED + FIXED):
    1) Nodes = endpoints + junctions
    2) Remove node regions from skeleton => segments
    3) For each segment:
       - find segment endpoints
       - if none, fallback to farthest-two pixels in segment
       - snap endpoint candidates to nearest nodes within endpoint_snap
       - choose BEST TWO DISTINCT nodes (closest snaps)
    """
    sk = _to01(skel)
    nodes = list(endpoints) + list(junctions)

    node_lab, node_mask = label_node_regions_voronoi(nodes, sk.shape, radius=node_radius)

    sk_wo_nodes = sk.copy()
    sk_wo_nodes[node_mask == 1] = 0

    num, labels = cv2.connectedComponents(sk_wo_nodes, connectivity=8)

    edges = []
    snap_r2 = float(endpoint_snap * endpoint_snap)

    for seg_id in range(1, num):
        ys, xs = np.where(labels == seg_id)
        if len(xs) < min_seg_area:
            continue

        seg_pixels = list(zip(xs.tolist(), ys.tolist()))
        if len(seg_pixels) < min_edge_pixels:
            continue

        seg_img = np.zeros_like(sk, dtype=np.uint8)
        seg_img[ys, xs] = 1

        seg_ends = segment_endpoints(seg_img)

        # Fallback for loops / no-end segments:
        if len(seg_ends) < 2:
            ends = farthest_two_points(seg_pixels)
            if ends is None:
                continue
            seg_ends = [ends[0], ends[1]]

        # Cap endpoints (but keep it large enough)
        if len(seg_ends) > max_endpoints_check:
            seg_ends = seg_ends[:max_endpoints_check]

        # Collect candidate snaps: (dist2, node_index)
        candidates = []
        for p in seg_ends:
            ni, d2 = nearest_node_for_point_with_dist(p, nodes)
            if ni is None:
                continue
            if d2 <= snap_r2:
                candidates.append((d2, ni))

        if len(candidates) < 2:
            continue

        candidates.sort(key=lambda t: t[0])

        # Pick two DISTINCT nodes
        chosen = []
        for _, ni in candidates:
            if ni not in chosen:
                chosen.append(ni)
            if len(chosen) == 2:
                break

        if len(chosen) < 2:
            continue

        u, v = sorted(chosen)
        if u == v:
            continue

        edges.append((u, v, seg_pixels))

    # dedupe edges (keep longest pixel list)
    uniq = {}
    for u, v, pix in edges:
        key = (u, v)
        if key not in uniq or len(pix) > len(uniq[key]):
            uniq[key] = pix

    edges_final = [(u, v, uniq[(u, v)]) for (u, v) in uniq.keys()]

    return nodes, edges_final, (sk_wo_nodes * 255).astype(np.uint8)


# ------------------ DRAW DEBUG ------------------

def draw_graph_debug(skel, nodes, edges):
    sk = _to01(skel) * 255
    vis = cv2.cvtColor(sk, cv2.COLOR_GRAY2BGR)

    for (u, v, pix) in edges:
        for (x, y) in pix:
            vis[y, x] = (0, 255, 0)

    for i, (x, y) in enumerate(nodes):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return vis


# ------------------ NODE MERGE (CLUSTER) ------------------

def merge_close_nodes(nodes, merge_radius=30):
    pts = np.array([(float(x), float(y)) for (x, y) in nodes], dtype=np.float32)
    n = len(pts)
    if n == 0:
        return [], []

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    r2 = float(merge_radius * merge_radius)
    for i in range(n):
        dx = pts[i, 0] - pts[:, 0]
        dy = pts[i, 1] - pts[:, 1]
        dist2 = dx * dx + dy * dy
        close = np.where(dist2 <= r2)[0]
        for j in close:
            union(i, int(j))

    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    merged_nodes = []
    mapping = [-1] * n

    for new_id, idxs in enumerate(clusters.values()):
        cx = float(np.mean(pts[idxs, 0]))
        cy = float(np.mean(pts[idxs, 1]))
        merged_nodes.append((cx, cy))
        for old_i in idxs:
            mapping[old_i] = new_id

    return merged_nodes, mapping


def remap_edges(edges, mapping):
    uniq = {}
    for u, v, pix in edges:
        u2, v2 = mapping[u], mapping[v]
        if u2 == v2:
            continue
        if u2 > v2:
            u2, v2 = v2, u2
        key = (u2, v2)
        if key not in uniq or len(pix) > len(uniq[key]):
            uniq[key] = pix
    return [(u, v, uniq[(u, v)]) for (u, v) in uniq.keys()]


# ------------------ DEGREE-2 COMPRESSION ------------------

def compress_degree2(nodes, edges):
    n = len(nodes)
    if n == 0 or len(edges) == 0:
        return nodes, edges

    edge_pix = {}
    for u, v, pix in edges:
        if u > v:
            u, v = v, u
        edge_pix[(u, v)] = list(pix)

    removed = set()
    changed = True

    while changed:
        changed = False

        adj = {i: set() for i in range(n)}
        for (u, v) in edge_pix.keys():
            adj[u].add(v)
            adj[v].add(u)

        for k in range(n):
            if k in removed:
                continue
            if len(adj[k]) == 2:
                a, b = list(adj[k])

                u1, v1 = (a, k) if a < k else (k, a)
                u2, v2 = (b, k) if b < k else (k, b)

                pix1 = edge_pix.get((u1, v1), [])
                pix2 = edge_pix.get((u2, v2), [])

                edge_pix.pop((u1, v1), None)
                edge_pix.pop((u2, v2), None)

                uu, vv = (a, b) if a < b else (b, a)
                merged = pix1 + pix2

                if (uu, vv) in edge_pix:
                    if len(merged) > len(edge_pix[(uu, vv)]):
                        edge_pix[(uu, vv)] = merged
                else:
                    edge_pix[(uu, vv)] = merged

                removed.add(k)
                changed = True
                break

    keep = [i for i in range(n) if i not in removed]
    new_index = {old: new for new, old in enumerate(keep)}

    new_nodes = [nodes[i] for i in keep]
    new_edges = []
    for (u, v), pix in edge_pix.items():
        if u in removed or v in removed:
            continue
        new_edges.append((new_index[u], new_index[v], pix))

    return new_nodes, new_edges
