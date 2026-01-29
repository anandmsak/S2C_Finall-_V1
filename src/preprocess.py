import cv2
import numpy as np
from skimage.morphology import skeletonize
import os

def extract_skeleton(image_path, save_dir="data/skeleton"):
    # 1. Read and validate image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Image not found at {image_path}")

    # 2. Isolate Blue Ink (HSV masking is robust for pen sketches)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 3. Morphological Cleanup (Connect small gaps in ink)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # 4. Skeletonize (True 1-pixel wide line using verified logic)
    skeleton = skeletonize(mask > 0)
    skeleton = (skeleton * 255).astype(np.uint8)

    # 5. Save results
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, "skeleton.png"), skeleton)
    
    return skeleton