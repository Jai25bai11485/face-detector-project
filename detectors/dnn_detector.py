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

class DNNDetector:
    """Face detector using a Caffe SSD deep neural network."""

    def __init__(self, confidence_threshold: float = 0.5):
        if not os.path.isfile(_PROTOTXT) or not os.path.isfile(_CAFFEMODEL):
            raise FileNotFoundError(
                "DNN model files not found.  Run:\n"
                "    python download_models.py\n"
                f"Expected location: {_MODELS_DIR}"
            )

        self.net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame, conf_threshold: float | None = None):
        """Detect faces in *frame* (BGR image).

        parameters:-

        frame : np.ndarray
            Input image in BGR colour space.
        conf_threshold : float, optional
            Overide the default confidence threshold for this call.
        """
        threshold = conf_threshold if conf_threshold is not None else self.confidence_threshold
        h, w = frame.shape[:2]

        # Pre-process: resize to 300×300 as expected by the SSD model
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),  # ImageNet BGR mean
        )

        self.net.setInput(blob)
        raw_detections = self.net.forward()

        detections = []
        for i in range(raw_detections.shape[2]):
            confidence = float(raw_detections[0, 0, i, 2])
            if confidence < threshold:
                continue

            # The network outputs normalised box coordinates
            box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            detections.append((x1, y1, x2 - x1, y2 - y1, confidence))

        return detections
