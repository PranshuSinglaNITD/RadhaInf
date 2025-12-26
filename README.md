**Project Overview**
- **Purpose:** This repository implements foggy-condition vehicle detection and classification (car, truck, bus, etc.) with a dehazing pre-process and alerting (visual + beep) on close/proximate vehicles.
- **Repo root:** contains training scripts, inference utilities, a small dehazing helper, and dataset manifests.

**Quick Features**
- **Detect & classify:** vehicle classes (bus, car, cycle, motorcycle, person, train, truck, van) as defined in data/foggy_dataset.yaml.
- **Dehazing:** simple CLAHE-based enhancement before detection (src/dehazing/dehaze.py).
- **Alerting:** non-blocking beep + visual warning when estimated distance < threshold (src/inference.py).
- **Modes:** run on single image, folder of images, video, or webcam.

**Performance**
- This project is designed and tuned to improve detection performance in foggy conditions: when the model is fine-tuned on foggy-domain data and augmented appropriately, it typically achieves higher recall and precision on foggy images than a vanilla off-the-shelf model that was not adapted for fog.
- Note: actual recall/precision gains depend on the training dataset, augmentation strategy, model size, and training schedule — evaluate on your test split and adjust hyperparameters accordingly.

**Demo on a video**
https://github.com/user-attachments/assets/19cbaef5-a3ca-4828-9316-a43dfff3232b

**Requirements**
- **Python:** 3.8+ recommended.
- **Install deps:**

```powershell
pip install -r requirements.txt
```

**Important packages:** `ultralytics`, `torch`, `opencv-python`, `albumentations`, `matplotlib` (fallback display)

**Key Files**
- **Train script:** train.py — trains YOLO model using `yolov8n.pt` base weights.
- **Inference CLI:** src/inference.py — run detection on images, folders, video, or webcam; supports `--show`, `--output`, `--no-beep`, and `--webcam`.
- **Detector wrapper:** src/detection/detector.py — wraps ultralytics YOLO model and filters allowed classes.
- **Dataset loader:** src/data/dataset.py — PyTorch Dataset for training pipeline and augmentations.
- **Dehaze helper:** src/dehazing/dehaze.py — simple CLAHE-based enhancement.
- **Config:** src/utils/config.py — thresholds and MODEL_PATH.

**Quick Start — Inference**
- Run on a single image (preview + save):

```powershell
python src/inference.py --input test_images/D040.png --show --output outputs/single_results
```

- Run on a folder of images (process & save annotated outputs):

```powershell
python src/inference.py --input test_images --show --output outputs/test_results
```

- Run webcam (explicit flag required) with preview:

```powershell
python src/inference.py --webcam --show --output outputs/webcam_results
```

- Headless mode (no preview, only save): omit `--show`.

**Quick Start — Training**
- The basic training call (uses ultralytics YOLO):

```powershell
python train.py
```

- Tips: train on GPU if available (remove `device='cpu'` from `train.py` or set `device=0`), increase `imgsz` and `epochs` for better performance, and consider switching from `yolov8n` to `yolov8s`/`yolov8m` for improved accuracy.

**Dataset format**
- The YAML manifest is at data/foggy_dataset.yaml. The repo expects images/labels layout under the paths defined in that file (default uses `data/foggy/...`).
- Labels use YOLO format: `class x_center y_center width height` (normalized).

**Configuration**
- Edit src/utils/config.py to adjust `CONFIDENCE_THRESHOLD`, `WARNING_DISTANCE_METERS`, and `MODEL_PATH`.
- You can override detection confidence at runtime by editing `CONFIDENCE_THRESHOLD` or by adding a debug/CLI option (future enhancement).

**Improving detection on foggy images (recommended workflow)**
- Short-term: lower `CONFIDENCE_THRESHOLD` (e.g. 0.15–0.25) for validation to surface weak detections.
- Preprocessing: experiment with stronger dehazing (AOD-Net, DCP) or stronger CLAHE parameters.
- Data: add labeled foggy images (the attached example) to training, or use synthetic fog augmentation (src/utils/augmentations.py uses RandomFog).
- Model: fine-tune a larger YOLO variant (yolov8s/yolov8m) and train longer with higher image sizes.

**Troubleshooting**
- If `cv2.imshow` fails with "The function is not implemented" on Windows, install GUI-enabled OpenCV wheel:

```powershell
pip uninstall opencv-python-headless opencv-python -y
pip install opencv-python
```

- If preview still fails, ensure `matplotlib` is installed; the code falls back to matplotlib for display.
- If inference is very slow, check `device` (CPU vs GPU) and model size; training on CPU is very slow — use GPU when possible.

**Next improvements / TODOs**
- Add CLI to override confidence without editing files.
- Add stronger dehaze module and benchmark improvement.
- Add pseudo-labeling script to expand training set from unlabeled foggy images.

**Contact / Maintainer**
- For changes, edit code in src/ and submit improvements via git. If you'd like, I can add debug flags, CI checks, or example notebooks to test detection on single images.

---
Generated: concise README for repository usage and quick troubleshooting.
