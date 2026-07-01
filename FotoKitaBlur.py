import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Load MediaPipe HandLandmarker
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=lambda result, image, timestamp: None
)

latest_result = None

def update_result(result, output_image, timestamp):
    global latest_result
    latest_result = result

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=update_result
)

detector = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(time.time() * 1000)
    detector.detect_async(mp_image, timestamp)

    peace = False

    if latest_result and latest_result.hand_landmarks:

        hand = latest_result.hand_landmarks[0]

        def finger_up(tip, pip):
            return hand[tip].y < hand[pip].y

        index = finger_up(8, 6)
        middle = finger_up(12, 10)
        ring = finger_up(16, 14)
        pinky = finger_up(20, 18)

        if index and middle and not ring and not pinky:
            peace = True

        # draw points
        for p in hand:
            x = int(p.x * frame.shape[1])
            y = int(p.y * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    if peace:
        frame = cv2.GaussianBlur(frame, (51, 51), 0)
        cv2.putText(frame, "PEACE DETECTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Foto Kita Blur", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()