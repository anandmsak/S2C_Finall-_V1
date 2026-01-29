import cv2
import os
import networkx as nx
from src.preprocess import extract_skeleton
from src.graph import skeleton_to_graph
from src.segment import segment_graph

# Configuration
IMAGE_PATH = "data/raw/input.png"

def run_s2c_pipeline():
    print("="*40)
    print("🔵 S2C: STARTING CIRCUIT RECONSTRUCTION")
    print("="*40)

    # Step 1: Image Processing
    print("[1/3] Preprocessing image...")
    skeleton = extract_skeleton(IMAGE_PATH)
    
    # Step 2: Graph Extraction
    print("[2/3] Building circuit graph...")
    G = skeleton_to_graph(skeleton)
    
    # Step 3: Segmentation (Wires vs Components)
    print("[3/3] Segmenting wires and components...")
    wire_graph, comp_graph = segment_graph(G)
   
    from src.segment import extract_component_images
    component_files = extract_component_images(comp_graph)
    print(f"✅ Extracted {len(component_files)} symbols stored ")
   
    # Output Stats
    print("\n" + "-"*20)
    print(f"📊 GRAPH RESULTS:")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    print(f"Wire Segments: {wire_graph.number_of_edges()}")
    print(f"Component Segments: {comp_graph.number_of_edges()}")
    print("-"*20)

    # Visualization
    cv2.imshow("Final Circuit Skeleton", skeleton)
    print("\n👉 Click the image and press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Put your blue-ink sketch in {IMAGE_PATH}")
    else:
        run_s2c_pipeline()