from ultralytics import YOLO
import cv2

MODEL_PATH = "models/best.pt"
IMAGE_PATH = "data/raw/input.png"

model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH)

r = results[0]
print("names:", r.names)
print("boxes xyxy:", r.boxes.xyxy[:5])
print("conf:", r.boxes.conf[:5])
print("cls:", r.boxes.cls[:5])
print("num boxes:", len(r.boxes))
