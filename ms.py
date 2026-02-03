'''
import cv2
import numpy
import skimage
import networkx
import shapely
import matplotlib

print("ALL OK")


import cv2
print(hasattr(cv2.ximgproc, "thinning"))
'''
import cv2
import numpy as np

# Test 1: Attribute Check
print("🔍 Checking for ximgproc attribute...")
if hasattr(cv2, 'ximgproc'):
    print("✅ SUCCESS: cv2.ximgproc is available.")
else:
    print("❌ ERROR: cv2.ximgproc NOT found. You need opencv-contrib-python.")

# Test 2: Execution Check
try:
    print("\n🔍 Testing thinning function execution...")
    # Create a dummy white square on black background
    dummy = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(dummy, (20, 20), (80, 80), 255, -1)
    
    # Attempt thinning
    result = cv2.ximgproc.thinning(dummy)
    print("✅ SUCCESS: Thinning function executed without crashing.")
    
except AttributeError as e:
    print(f"❌ CRASH: AttributeError - {e}")
except Exception as e:
    print(f"❌ CRASH: Unexpected Error - {type(e).__name__}: {e}")

print("\n🚀 Diagnostic complete.")