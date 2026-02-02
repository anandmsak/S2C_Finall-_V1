import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("../models/best.pt")


CONF = 0.18  # confidence threshold

cap = cv2.VideoCapture(0)  # change to 1 if external webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=CONF, imgsz=640, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("S2C - Component Detection", annotated)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
