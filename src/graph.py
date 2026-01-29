# src/graph.py
import cv2
import numpy as np
import networkx as nx

NEIGHBORS = [(-1,-1), (-1,0), (-1,1),
             (0,-1),          (0,1),
             (1,-1),  (1,0),  (1,1)]

def get_neighbors(img, y, x):
    h, w = img.shape
    pts = []
    for dy, dx in NEIGHBORS:
        ny, xx = y + dy, x + dx
        if 0 <= ny < h and 0 <= xx < w and img[ny, xx] > 0:
            pts.append((ny, xx))
    return pts
def cluster_nodes(nodes, radius=10):
    clusters = []
    nodes = list(nodes)

    while nodes:
        base = nodes.pop(0)
        cluster = [base]

        for p in nodes[:]:
            if abs(p[0] - base[0]) <= radius and abs(p[1] - base[1]) <= radius:
                cluster.append(p)
                nodes.remove(p)

        clusters.append(cluster)

    # return cluster center
    return [(
        int(np.mean([p[0] for p in c])),
        int(np.mean([p[1] for p in c]))
    ) for c in clusters]

# src/graph.py

# src/graph.py

# src/graph.py

# src/graph.py

# src/graph.py

def prune_graph(G):
    """
    Adaptive Pruning: Uses average edge length to remove noise and 
    consolidate broken wire segments.
    """
    if G.number_of_edges() == 0: return G
    
    # Calculate adaptive threshold: 30% of the average edge length
    avg_len = np.mean([data['length'] for u, v, data in G.edges(data=True)])
    hair_threshold = max(20, avg_len * 0.3) 

    # 1. Remove 'Hairs' (Short dead-ends)
    for _ in range(2):
        to_remove = [n for n in G.nodes() if G.degree(n) == 1 and 
                     any(G[n][neigh]['length'] < hair_threshold for neigh in G.neighbors(n))]
        G.remove_nodes_from(to_remove)

    # 2. Path Consolidation (Fixes the '11 wire segments' problem)
    # Merges all degree-2 nodes until only true junctions/endpoints remain.
    while True:
        degree_2_nodes = [n for n in G.nodes() if G.degree(n) == 2]
        if not degree_2_nodes: break
        for n in degree_2_nodes:
            if n in G:
                neighs = list(G.neighbors(n))
                u, v = neighs[0], neighs[1]
                new_len = G[u][n]['length'] + G[n][v]['length']
                new_pix = G[u][n]['pixels'] + G[n][v]['pixels']
                G.add_edge(u, v, length=new_len, pixels=new_pix)
                G.remove_node(n)

    # 3. Component/Junction Merge
    # Collapses nodes within 50 pixels to handle hand-drawn junction jitter.
    nodes = list(G.nodes())
    for i, n1 in enumerate(nodes):
        for n2 in nodes[i+1:]:
            if n1 in G and n2 in G:
                dist = np.sqrt((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2)
                if dist < 50: 
                    G = nx.contracted_nodes(G, n1, n2, self_loops=False)
    return G

def skeleton_to_graph(skeleton):
    
    # remove tiny isolated pixels
    #kernel = np.ones((3,3), np.uint8)
    #skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)

    G = nx.Graph()
    h, w = skeleton.shape

    degree_map = {}
    for y in range(h):
        for x in range(w):
            if skeleton[y, x] > 0:
                degree_map[(y, x)] = len(get_neighbors(skeleton, y, x))

        raw_nodes = {p for p, d in degree_map.items() if d > 2 or d == 1}
        nodes = cluster_nodes(raw_nodes, radius=10)      

    visited = set()

    for node in nodes:
        G.add_node(node)

    for node in nodes:
        y, x = node
        for ny, xx in get_neighbors(skeleton, y, x):
            if (node, (ny, xx)) in visited:
                continue

            path = [node]
            py, px = y, x
            cy, cx = ny, xx

            while True:
                path.append((cy, cx))
                visited.add(((py, px), (cy, cx)))
                visited.add(((cy, cx), (py, px)))

                if (cy, cx) in nodes and len(path) > 5:
                    G.add_edge(node, (cy, cx), pixels=path.copy(), length=len(path))
                    break

                neigh = get_neighbors(skeleton, cy, cx)
                next_pts = [p for p in neigh if p != (py, px)]
                if not next_pts:
                    break

                py, px = cy, cx
                cy, cx = next_pts[0]
   
    G = prune_graph(G)
    return G
