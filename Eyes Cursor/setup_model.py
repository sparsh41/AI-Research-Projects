import urllib.request
import os
import sys


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def download_model():
    if os.path.exists(MODEL_PATH):
        return

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print(f"Failed to download model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()