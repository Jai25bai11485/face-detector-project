"""
Face Detection App — CLI Entry Point

Usage:
    python main.py --mode webcam  --method haar
    python main.py --mode webcam  --method dnn
    python main.py --mode webcam  --method both
    python main.py --mode image   --method dnn  --input photo.jpg

Controls (in the display window):
    q   — quit
"""

import argparse
import sys
import cv2

from detectors.haar_detector import HaarDetector
from detectors.dnn_detector import DNNDetector
from utils import draw_detections, resize_frame


# Colour palette
HAAR_COLOR = (0, 255, 128)
DNN_COLOR  = (255, 128, 0)
BOTH_HAAR  = (0, 255, 128)
BOTH_DNN   = (255, 128, 0)


# Helpers

def build_detectors(method: str, confidence: float):
    """Return a list of ``(detector, label, colour)`` tuples."""
    detectors = []
    if method in ("haar", "both"):
        detectors.append((HaarDetector(), "Haar", HAAR_COLOR))
    if method in ("dnn", "both"):
        detectors.append((DNNDetector(confidence_threshold=confidence), "DNN", DNN_COLOR))
    return detectors


def annotate(frame, detectors):
    """Run every detector and draw results on *frame*."""
    y_offset = 30

    for det, label, color in detectors:
        faces = det.detect(frame)
        count = len(faces)
        draw_detections(frame, faces, label=label, color=color)

        # Draw individual face count for this detector
        text = f"{label} Faces: {count}"
        cv2.putText(
            frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        y_offset += 30  # Move down for the next detector's text

    return frame

# Modes

def run_webcam(detectors, cam_index: int = 0):
    """Live webcam detection loop."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.  Check your camera index.", file=sys.stderr)
        sys.exit(1)

    print("Webcam opened.  Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.", file=sys.stderr)
            break

        frame = resize_frame(frame, width=720)
        annotate(frame, detectors)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(detectors, image_path: str):
    """Detect faces in a single image file."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}", file=sys.stderr)
        sys.exit(1)

    frame = resize_frame(frame, width=720)
    annotate(frame, detectors)

    cv2.imshow("Face Detection", frame)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Face detection with Haar Cascades and/or DNN (Caffe SSD)."
    )
    parser.add_argument(
        "--mode",
        choices=["webcam", "image"],
        default="webcam",
        help="Input source (default: webcam).",
    )
    parser.add_argument(
        "--method",
        choices=["haar", "dnn", "both"],
        default="both",
        help="Detection method (default: both).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to image file (required when --mode image).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for DNN detector (default: 0.5).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "image" and args.input is None:
        print("[ERROR] --input is required when --mode is 'image'.", file=sys.stderr)
        sys.exit(1)

    print(f"Mode   : {args.mode}")
    print(f"Method : {args.method}")
    print(f"Conf.  : {args.confidence}")
    print()

    detectors = build_detectors(args.method, args.confidence)

    if args.mode == "webcam":
        run_webcam(detectors, cam_index=args.camera)
    else:
        run_image(detectors, args.input)


if __name__ == "__main__":
    main()