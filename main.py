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


