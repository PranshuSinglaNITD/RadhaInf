from ultralytics import YOLO
import torch
torch.set_num_threads(4)

model = YOLO("yolov8s.pt")

model.train(
    data="data/foggy_dataset.yaml",
    epochs=20,
    imgsz=512,
    batch=8,
    device='cpu',
    augment=True
)
