import albumentations as A
import cv2

def get_train_augmentations():
    """
    Augmentations focused on foggy driving conditions
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),

            # Fog / visibility simulation
            A.RandomFog(
                fog_coef_lower=0.5,
                fog_coef_upper=0.9,
                alpha_coef=0.12,
                p=0.7
            ),

            # Low contrast (common in fog)
            A.RandomBrightnessContrast(
                brightness_limit=-0.2,
                contrast_limit=-0.3,
                p=0.5
            ),

            # Blur due to moisture / camera focus
            A.GaussianBlur(blur_limit=(3,7), p=0.3),

            # Sensor noise
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

            # Resize for YOLO
            A.Resize(640, 640)
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"]
        )
    )
