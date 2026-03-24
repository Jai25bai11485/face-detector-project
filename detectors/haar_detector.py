import cv2


class HaarDetector:


    def __init__(self):
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {cascade_path}"
            )

    def detect(self, frame):
        """Detect faces in *frame* (BGR image)."""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # detectMultiScale3 gives us reject-level weights we can use
        # as a proxy for confidence.
        faces, reject_levels, level_weights = self.classifier.detectMultiScale3(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=0,
            outputRejectLevels=True,
        )

        detections = []
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):


                if i < len(level_weights):
                    w_val = level_weights[i]
                    weight = float(w_val.flat[0]) if hasattr(w_val, 'flat') else float(w_val)
                else:
                    weight = 0.0
                confidence = min(weight / 5.0, 1.0)
                detections.append((int(x), int(y), int(w), int(h), confidence))

        return detections
