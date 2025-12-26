import cv2
import os
from torch.utils.data import Dataset
from src.utils.augmentations import get_train_augmentations

class FoggyVehicleDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        # only include common image extensions
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = get_train_augmentations()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # handle any image extension when mapping to label file
        label_path = os.path.splitext(os.path.join(self.label_dir, img_name))[0] + ".txt"

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []

        # if label file missing, return empty labels
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = map(float, parts[:5])
                        bboxes.append([x, y, w, h])
                        class_labels.append(int(cls))

        augmented = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        return augmented["image"], augmented["bboxes"], augmented["class_labels"]
