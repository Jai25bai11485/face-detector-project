import os
import urllib.request
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, url in FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"[SKIP] {filename} already exists.")
            continue

        print(f"[DOWNLOADING] {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[OK]   {filename}  ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print("\nAll model files are ready in:", MODELS_DIR)


if __name__ == "__main__":
    download_models()
