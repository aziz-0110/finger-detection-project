import cv2
import mediapipe as mp
import numpy as np
from draw1 import *
# from cvzone.HandTrackingModule import HandDetector
# import cvzone

src = "dataset.mp4"
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

prev_x, prev_y, canvas = None, None, None

while True:
    ret, frame = cap.read()

    if not ret: break

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    detect = hands.process(frame)
    if canvas is None:
        # canvas = frame.copy()
        canvas = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
    if detect.multi_hand_landmarks:
        for handType, handLms in zip(detect.multi_handedness, detect.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            if handType.classification[0].label == handType.classification[0].label:  # masih blm berguna
                kanan = detect.multi_hand_landmarks[0]
                telunjuk = kanan.landmark[8]
                tengah = kanan.landmark[12]
                h, w = frame.shape[:2]
                x = int(telunjuk.x * w)
                y = int(telunjuk.y * h)

                if telunjuk.y < tengah.y:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    if prev_x != None and prev_y != None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), thickness=5)

                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = x, y

    overlay_frame = np.zeros_like(canvas, dtype=np.uint8)
    overlay_frame[ :, :, :3] = frame
    overlay_frame[ :, :, 3] = 255

    # cv2.rectangle(overlay_frame, (5, 15), (220, 120), (255, 0, 0), 2)

    comb_frame = cv2.addWeighted(overlay_frame, 1, canvas, 0.5, 0)

    cv2.imshow("bb", canvas)
    cv2.imshow("aa", comb_frame)

    key = cv2.waitKey(1)

    key_actions = {
        ord("j"): lambda: drawing(canvas, 1),
        ord("k"): lambda: drawing(canvas, 2),
        ord("l"): lambda: drawing(canvas, 3),
        ord("s"): lambda: cv2.imwrite("result.jpg", canvas),
        ord("w"): lambda: (None, None, None),
    }

    if key in key_actions:
        action = key_actions[key]()
        if key in [ord("j"), ord("k"), ord("l")]:
            canvas[ :, :, :3] = action

    if key == ord("c"):
        canvas, prev_x, prev_y = None, None, None
    if key == ord("q"): break

cap.release()
cv2.destroyAllWindows()
