# src/segment.py
import math
import cv2
import os
import numpy as np
import networkx as nx

def edge_angle(edge_pixels):
    (y1, x1) = edge_pixels[0]
    (y2, x2) = edge_pixels[-1]
    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    return min(angle, 180 - angle)

def segment_graph(G, wire_length_thresh=40):
    wire_edges = []
    component_edges = []

    for u, v, data in G.edges(data=True):
        length = data["length"]
        angle = edge_angle(data["pixels"])

        if length >= wire_length_thresh and (angle < 15 or abs(angle - 90) < 15):
            wire_edges.append((u, v))
        else:
            component_edges.append((u, v))

    wire_graph = G.edge_subgraph(wire_edges).copy()
    comp_graph = G.edge_subgraph(component_edges).copy()

    return wire_graph, comp_graph

def extract_component_images(comp_graph, save_dir="data/components"):
    """
    Tightly crops component symbols by filtering out wire 'tails' far from the center.
    """
    import networkx as nx
    os.makedirs(save_dir, exist_ok=True)
    components = list(nx.connected_components(comp_graph))
    saved_files = []

    for i, nodes in enumerate(components):
        subgraph = comp_graph.subgraph(nodes)
        all_pixels = []
        for u, v, data in subgraph.edges(data=True):
            all_pixels.extend(data['pixels'])
        
        if not all_pixels: continue

        # 1. FIND THE CORE: Symbols (zigzag/plates) have higher pixel density
        pts = np.array(all_pixels)
        center_y, center_x = np.mean(pts, axis=0)
        
        # 2. RADIUS CROP: Only keep pixels within 50px of the center
        # This removes the 'L' shapes and long wires seen in your uploaded images.
        dist = np.sqrt((pts[:,0] - center_y)**2 + (pts[:,1] - center_x)**2)
        core_pts = pts[dist < 50] 

        if len(core_pts) < 10: core_pts = pts # Safety fallback

        # 3. BOUNDING BOX & SAVE
        y_min, x_min = core_pts.min(axis=0)
        y_max, x_max = core_pts.max(axis=0)
        
        pad = 10
        h, w = (y_max - y_min + 2*pad), (x_max - x_min + 2*pad)
        blob_img = np.zeros((h, w), dtype=np.uint8)
        
        for y, x in core_pts:
            blob_img[int(y - y_min + pad), int(x - x_min + pad)] = 255
            
        file_path = os.path.join(save_dir, f"comp_{i}.png")
        cv2.imwrite(file_path, blob_img)
        saved_files.append(file_path)
        print(f"🎯 Cleaned symbol saved: {file_path}")

    return saved_files