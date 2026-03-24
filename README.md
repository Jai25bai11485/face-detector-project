#  Face Detection App

A real-time face detection application built with Python and OpenCV. Supports two distinct detection strategies — classical **Haar Cascade** and deep-learning-based **Caffe SSD DNN** — which can be run individually or side-by-side for comparison. Works on live webcam feeds or static images.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [File Reference](#file-reference)
- [How the Files Interact](#how-the-files-interact)
- [Detection Methods Explained](#detection-methods-explained)
- [Configuration & Tuning](#configuration--tuning)

---

##  Features

- **Live webcam detection** with real-time bounding boxes and face counts
- **Static image detection** for single-frame analysis
- **Two detection methods** available independently or simultaneously:
  - Haar Cascade (fast, CPU-friendly, classical CV)
  - DNN / Caffe SSD (more accurate, deep learning-based)
- **Per-detector face count** overlaid on the display window
- **Confidence scores** shown on each bounding box
- **Colour-coded annotations** to visually distinguish the two detectors
- **Aspect-ratio-preserving resizing** for consistent display

---

##  Requirements

- Python 3.10+
- OpenCV with DNN support: `opencv-python` or `opencv-contrib-python`
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

---

##  Installation

```bash
git clone https://github.com/your-username/face-detection-app.git
cd face-detection-app
pip install -r requirements.txt
```

---

##  Model Setup

The DNN detector requires two Caffe model files which are **not bundled in the repository** due to their size. Download them before using the `dnn` or `both` methods:

```bash
python download_models.py
```

 


The Haar Cascade XML file (`haarcascade_frontalface_default.xml`) ships with OpenCV and is loaded automatically — no extra download needed.

---

##  Usage

### Webcam — both detectors (default)

```bash
python main.py
```

### Webcam — Haar only

```bash
python main.py --mode webcam --method haar
```

### Webcam — DNN only, custom confidence threshold

```bash
python main.py --mode webcam --method dnn --confidence 0.6
```

### Static image — DNN detector

```bash
python main.py --mode image --method dnn --input photo.jpg
```

### Alternate webcam device

```bash
python main.py --mode webcam --camera 1
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit the display window |
| Any key | Close window (image mode only) |

### Full CLI Reference

```
usage: main.py [-h] [--mode {webcam,image}] [--method {haar,dnn,both}]
               [--input INPUT] [--confidence CONFIDENCE] [--camera CAMERA]

optional arguments:
  --mode        Input source: webcam or image (default: webcam)
  --method      Detection method: haar, dnn, or both (default: both)
  --input       Path to image file — required when --mode is image
  --confidence  Minimum DNN confidence threshold, 0–1 (default: 0.5)
  --camera      Webcam device index (default: 0)
```

---

##  Repository Structure

```
face-detection-app/
│
├── main.py                   # CLI entry point — orchestrates all modes and detectors
│
├── utils.py                  # Shared drawing and frame-resize utilities
│
├── detectors/                # Detector package
│   ├── __init__.py           # Package marker
│   ├── haar_detector.py      # Haar Cascade detector class
│   └── dnn_detector.py       # Caffe SSD DNN detector class
│
├── models/                   # Pre-trained model files (downloaded separately)
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── download_models.py        # Helper script to fetch model files
└── requirements.txt
```

---

## File Reference

### `main.py` — Entry Point & Orchestrator

The top-level script. Responsible for:

- **Parsing CLI arguments** via `argparse` (`--mode`, `--method`, `--input`, `--confidence`, `--camera`)
- **Instantiating detectors** through `build_detectors()`, which returns a list of `(detector, label, colour)` tuples based on the chosen method
- **Running the appropriate mode** — `run_webcam()` for live capture or `run_image()` for a static file
- **Calling `annotate()`** on every frame, which loops over all active detectors, collects bounding boxes, delegates drawing to `utils.draw_detections()`, and overlays per-detector face counts



---

### `utils.py` — Drawing & Display Utilities

A stateless helper module with two functions:

#### `draw_detections(frame, detections, label, color)`

Iterates over a list of `(x, y, w, h, confidence)` tuples and draws on the frame **in-place**:

1. A coloured rectangle around the detected face region
2. A filled colour background patch above the box
3. A black text label showing `"<Label>: <confidence%>"` on that background

#### `resize_frame(frame, width=720)`

Resizes a frame to the target `width` while preserving the original aspect ratio. Returns the frame unchanged if it already matches the target width.

---

### `detectors/__init__.py` — Package Marker

An empty (single comment) file that tells Python to treat the `detectors/` directory as an importable package. No logic lives here.

---

### `detectors/haar_detector.py` — Haar Cascade Detector

Implements the `HaarDetector` class.

**`__init__()`**
Loads OpenCV's bundled `haarcascade_frontalface_default.xml` using `cv2.CascadeClassifier`. Raises a `RuntimeError` if the file cannot be loaded.

**`detect(frame) → list[tuple]`**

Processing pipeline:
1. Convert the frame to **grayscale**
2. Apply **histogram equalisation** (`cv2.equalizeHist`) to normalise lighting conditions
3. Run `detectMultiScale3` with `outputRejectLevels=True` to obtain raw detection weights alongside face rectangles
4. **Normalise weights** into a `[0.0, 1.0]` confidence range by clamping `weight / 5.0`
5. Return a list of `(x, y, w, h, confidence)` tuples



---

### `detectors/dnn_detector.py` — Caffe SSD DNN Detector

Implements the `DNNDetector` class using OpenCV's `dnn` module with a ResNet-10-based Single Shot Multibox Detector (SSD) pre-trained on face data.

**`__init__(confidence_threshold=0.5)`**
Verifies both model files exist in `models/`, then loads the network via `cv2.dnn.readNetFromCaffe()`. Raises `FileNotFoundError` with download instructions if files are missing.

**`detect(frame, conf_threshold=None) → list[tuple]`**

Processing pipeline:
1. **Pre-process** the frame into a 300×300 blob, subtracting the ImageNet BGR mean `(104.0, 177.0, 123.0)`
2. Run a **forward pass** through the network via `self.net.forward()`
3. Iterate raw detections; skip any with confidence below the threshold
4. **De-normalise** bounding box coordinates by multiplying by `[w, h, w, h]`
5. **Clamp** coordinates to frame boundaries to prevent out-of-bounds boxes
6. Return `(x1, y1, width, height, confidence)` tuples

The `conf_threshold` parameter on `detect()` allows per-call overrides of the instance-level threshold set at construction.

---

##  How the Files Interact

```
main.py
  │
  ├── parse_args()
  │       └── argparse → mode, method, confidence, camera, input
  │
  ├── build_detectors(method, confidence)
  │       ├── HaarDetector()          ← detectors/haar_detector.py
  │       └── DNNDetector(confidence) ← detectors/dnn_detector.py
  │
  ├── run_webcam(detectors, cam_index)   OR   run_image(detectors, path)
  │       │
  │       └── annotate(frame, detectors)
  │               │
  │               ├── det.detect(frame)          ← each detector's detect()
  │               │       returns [(x,y,w,h,conf), ...]
  │               │
  │               └── draw_detections(frame, faces, label, color)
  │                       └── utils.py           ← draws boxes + labels in-place
  │
  └── resize_frame(frame, width=720)     ← utils.py
```

**Data flow in detail:**

1. `main.py` reads CLI arguments and constructs the list of active detectors.
2. For each video frame (or the single image), `resize_frame()` from `utils.py` standardises dimensions to 720 px wide.
3. `annotate()` iterates over every detector tuple. It calls `det.detect(frame)` which returns a list of bounding-box tuples — the same `(x, y, w, h, confidence)` shape regardless of which detector produced them. This **uniform interface** means `main.py` and `utils.py` are completely agnostic about which algorithm ran.
4. `draw_detections()` in `utils.py` takes that list and renders boxes and labels onto the frame in-place.
5. A per-detector face count is written onto the frame with a colour-coded `cv2.putText` call in `annotate()`.
6. The annotated frame is displayed via `cv2.imshow`.

---

##  Detection Methods Explained

### Haar Cascade

A classical sliding-window method. A cascade of simple feature classifiers (edge, line, and rectangle features) trained with AdaBoost rapidly rejects non-face regions and focuses computation on likely face areas. It is very fast on CPU but sensitive to pose, lighting, and partial occlusion.

### Caffe SSD DNN (ResNet-10)

A Single Shot Multibox Detector with a lightweight ResNet-10 backbone, trained on face imagery. It processes the whole image in a single forward pass at 300×300 resolution, outputting confidence scores and normalised bounding boxes directly. It is more robust to varied angles, lighting, and partial faces, but requires the model files to be present and is slightly slower than Haar on CPU-only hardware.

---

