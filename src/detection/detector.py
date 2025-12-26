from ultralytics import YOLO
import cv2
from src.utils.config import CONFIDENCE_THRESHOLD

class VehicleDetector:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load model '{model_path}': {e}")
            raise
        # use lowercase class names for comparison
        self.allowed_classes = ["car", "truck", "bus"]

    def detect(self, frame):
        detections = []
        try:
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD)
        except Exception as e:
            print(f"Model inference error: {e}")
            return detections

        for r in results:
            for box in getattr(r, 'boxes', []):
                try:
                    cls_id = int(box.cls[0])
                    label = self.model.names.get(cls_id, str(cls_id)) if isinstance(self.model.names, dict) else self.model.names[cls_id]
                    # compare in lowercase to avoid case mismatches
                    if label.lower() in self.allowed_classes:
                        coords = box.xyxy[0]
                        x1, y1, x2, y2 = map(int, coords)
                        detections.append((label, x1, y1, x2, y2))
                except Exception:
                    continue

        return detections
