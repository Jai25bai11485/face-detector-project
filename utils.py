"""
Shared drawing and display utilities.
"""

import cv2


def draw_detections(frame, detections, label="Face", color=(0, 255, 0)):
    """Draw bounding boxes and confidence labels on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (modified in-place).
    detections : list[tuple[int, int, int, int, float]]
        List of ``(x, y, w, h, confidence)`` tuples.
    label : str
        Prefix text shown above each box (e.g. "Haar", "DNN").
    color : tuple[int, int, int]
        BGR colour for the box and text.
    """
    for x, y, w, h, conf in detections:
        # Rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        text = f"{label}: {conf:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame, (x, y - th - 10), (x + tw + 4, y), color, cv2.FILLED
        )

        # Label text
        cv2.putText(
            frame,
            text,
            (x + 2, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return frame


def resize_frame(frame, width=720):
    """Resize *frame* to the given *width* while keeping aspect ratio."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    ratio = width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h))
