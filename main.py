import cv2
import os
from src.preprocess import extract_skeleton
from src.graph import skeleton_to_graph

# Configuration
IMAGE_PATH = "data/raw/input2.png"


def run_s2c_pipeline():
    print("=" * 40)
    print("🔵 S2C: STARTING CIRCUIT RECONSTRUCTION")
    print("=" * 40)

    # Step 1: Preprocessing
    print("[1/3] Preprocessing image...")
    wire_skel, components = extract_skeleton(IMAGE_PATH)

    # ---- DEBUG VISUALIZATION (MANDATORY FOR NOW) ----
    img = cv2.imread(IMAGE_PATH)
    for c in components:
        x, y, w, h = c["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Detected Components", img)
    cv2.imshow("Wire Skeleton", wire_skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # ------------------------------------------------

    # Step 2: Graph Extraction (WIRES ONLY)
    print("[2/3] Building circuit graph...")
    G = skeleton_to_graph(wire_skel)

    # Output Stats
    print("\n" + "-" * 20)
    print("📊 GRAPH RESULTS:")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    print("-" * 20)


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Put your blue-ink sketch in {IMAGE_PATH}")
    else:
        run_s2c_pipeline()
