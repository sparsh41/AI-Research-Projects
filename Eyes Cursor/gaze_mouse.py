import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import sys

# Constants
SMOOTHING_FACTOR = 4
BLINK_THRESHOLD = 0.004
CLICK_THRESHOLD_FRAMES = 15
CAMERA_INDEX = 0


MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'face_landmarker.task')

LEFT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

def get_iris_center(landmarks, iris_indices, frame_w, frame_h):
    x_coords = []
    y_coords = []

    for index in iris_indices:
        lm = landmarks[index]
        x_coords.append(lm.x * frame_w)
        y_coords.append(lm.y * frame_h)

    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))

    return center_x, center_y


def detect_blink(landmarks, top_idx, bottom_idx):

    top = landmarks[top_idx]
    bottom = landmarks[bottom_idx]

    distance = abs(top.y - bottom.y)

    return distance < BLINK_THRESHOLD

def map_to_screen(iris_x, iris_y, frame_w, frame_h, screen_w, screen_h):

    norm_x = iris_x / frame_w
    norm_y = iris_y / frame_h

    scale_x = 2.5
    scale_y = 2.5

    mapped_x = (norm_x - 0.5) * scale_x + 0.5
    mapped_y = (norm_y - 0.5) * scale_y + 0.5

    mapped_x = np.clip(mapped_x, 0, 1)
    mapped_y = np.clip(mapped_y, 0, 1)

    return int(mapped_x * screen_w), int(mapped_y * screen_h)


def main():

    if not os.path.exists(MODEL_PATH):
        sys.exit(1)

    # Get screen size

    screen_w, screen_h = pyautogui.size()
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True

    BaseOptions = mp.tasks.BaseOptions
    FaceLandMarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces = 1,
        min_face_detection_confidence = 0.5,
        min_face_presence_confidence = 0.5,
        min_tracking_confidence = 0.5
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not(cap.isOpened()):
        return

    smooth_x, smooth_y = screen_w // 2, screen_h // 2
    click_cooldown = 0
    last_timestamp_ms = 0

    with FaceLandMarker.create_from_options(options) as face_landmarker:

        while True:
            ret, frame = cap.read()
            if not ret:
                return

            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format= mp.ImageFormat.SRGB, data = rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                iris_x, iris_y = get_iris_center(landmarks, LEFT_IRIS, screen_w, screen_h)
                target_x, target_y = map_to_screen(iris_x, iris_y, frame_w, frame_h, screen_w, screen_h)

                smooth_x += (target_x - smooth_x) / SMOOTHING_FACTOR
                smooth_y += (target_y - smooth_y) / SMOOTHING_FACTOR

                try:
                    pyautogui.moveTo(int(smooth_x), int(smooth_y))
                except pyautogui.FailSafeException:
                    smooth_x, smooth_y = screen_w // 2, screen_h // 2

                if click_cooldown > 0:
                    click_cooldown -= 1

                if detect_blink(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM):
                    if click_cooldown == 0:
                        pyautogui.click()
                        click_cooldown = CLICK_THRESHOLD_FRAMES

                cv2.circle(frame, (iris_x, iris_y), 5, (0, 255, 0), -1)

            cv2.putText(
                frame, "q to quit",
                (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()