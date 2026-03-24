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

