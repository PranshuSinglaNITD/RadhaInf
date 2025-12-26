import argparse
import os
import glob
import cv2
import threading
from src.dehazing.dehaze import dehaze_image
from src.detection.detector import VehicleDetector
from src.utils.distance import estimate_distance
from src.utils.config import *

# optional matplotlib fallback if OpenCV was built without GUI support
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


def safe_imshow(winname, frame, block=False, wait_ms=1):
    try:
        cv2.imshow(winname, frame)
        return 'cv2'
    except Exception:
        if not _HAS_MATPLOTLIB:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.title(winname)
        plt.axis('off')
        if block:
            plt.show()
        else:
            plt.pause(max(wait_ms / 1000.0, 0.001))
            plt.draw()
        return 'plt'


# cross-platform beep
try:
    import winsound
    def _beep(freq=1000, dur=200):
        try:
            winsound.Beep(freq, dur)
        except Exception:
            pass
except Exception:
    def _beep(freq=1000, dur=200):
        print('\a')


def annotate_and_alert(frame, detections, beep=True):
    for label, x1, y1, x2, y2 in detections:
        box_height = y2 - y1
        distance = estimate_distance(box_height, frame.shape[0])

        color = (0, 255, 0)
        if distance < WARNING_DISTANCE_METERS:
            color = (0, 0, 255)
            cv2.putText(
                frame, "COLLISION RISK",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )
            if beep:
                threading.Thread(
                    target=_beep, args=(1000, 200), daemon=True
                ).start()

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {distance:.1f}m",
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )


def list_images(path):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(path, e)))
    files.sort()
    return files


def run_on_path(source, output_dir=None, show=False, beep=True):
    detector = VehicleDetector(MODEL_PATH)

    #video or webcam
    if (
        isinstance(source, int)
        or (isinstance(source, str) and source.startswith("http"))
        or (isinstance(source, str) and source.lower().endswith((".mp4", ".avi", ".mov")))
    ):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("❌ Cannot open video source")
            return

        writer = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = os.path.join(output_dir, "output.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640))
            enhanced = dehaze_image(frame)
            detections = detector.detect(enhanced)
            annotate_and_alert(frame, detections, beep=beep)

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("Foggy Vehicle Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        return

    # for images
    if os.path.isdir(source):
        images = list_images(source)
    elif os.path.isfile(source):
        images = [source]
    else:
        raise FileNotFoundError(f"Input path not found: {source}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 640))
        enhanced = dehaze_image(frame)
        detections = detector.detect(enhanced)
        annotate_and_alert(frame, detections, beep=beep)

        if output_dir:
            cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), frame)

        if show:
            cv2.imshow("Foggy Vehicle Detection", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break



def main():
    parser = argparse.ArgumentParser(
        description="Foggy vehicle detection on webcam, android camera, image, or folder."
    )

    parser.add_argument("--input", "-i", default=None,
                        help="Image file or directory")
    parser.add_argument("--webcam", "-w", action="store_true",
                        help="Use laptop webcam")
    parser.add_argument("--android-url", "-a", default=None,
                        help="Android IP camera URL (e.g. http://192.168.1.5:8080/video)")
    parser.add_argument("--output", "-o", default=None,
                        help="Directory to save outputs")
    parser.add_argument("--show", action="store_true",
                        help="Show output window")
    parser.add_argument("--no-beep", action="store_true",
                        help="Disable beep alert")

    args = parser.parse_args()

    if args.webcam:
        run_on_path(0, output_dir=args.output, show=args.show, beep=not args.no_beep)
        return

    if args.android_url:
        run_on_path(args.android_url, output_dir=args.output, show=args.show, beep=not args.no_beep)
        return

    if not args.input:
        parser.print_help()
        return

    run_on_path(args.input, output_dir=args.output, show=args.show, beep=not args.no_beep)



if __name__ == "__main__":
    main()
