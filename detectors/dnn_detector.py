import os
import cv2
import numpy as np


_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
_PROTOTXT = os.path.join(_MODELS_DIR, "deploy.prototxt")
_CAFFEMODEL = os.path.join(
    _MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel"
)

