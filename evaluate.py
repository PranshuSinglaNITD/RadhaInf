from ultralytics import YOLO

model = YOLO("runs/detect/train6/weights/best.pt")

metrics = model.val(
    data="data/foggy_dataset.yaml",
    split="test"
)

print("Precision:", metrics.box.p)
print("Recall:", metrics.box.r)
print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)
