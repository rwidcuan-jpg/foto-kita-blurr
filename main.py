import cv2
import mediapipe as mp
import numpy as np

# ==================================================
# MEDIAPIPE
# ==================================================

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# ==================================================
# CAMERA
# ==================================================

cap = cv2.VideoCapture(0)

# ==================================================
# HAND DETECTION
# ==================================================

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==================================================
# MAIN LOOP
# ==================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Simpan gambar asli
    original = frame.copy()

    # ==================================================
    # MEDIAPIPE HAND PROCESSING
    # ==================================================

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    two_finger_detected = False

    # ==================================================
    # CHECK FINGERS
    # ==================================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = hand_landmarks.landmark

            fingers = []

            # INDEX
            if landmarks[8].y < landmarks[6].y:
                fingers.append(1)
            else:
                fingers.append(0)

            # MIDDLE
            if landmarks[12].y < landmarks[10].y:
                fingers.append(1)
            else:
                fingers.append(0)

            # RING
            if landmarks[16].y < landmarks[14].y:
                fingers.append(1)
            else:
                fingers.append(0)

            # PINKY
            if landmarks[20].y < landmarks[18].y:
                fingers.append(1)
            else:
                fingers.append(0)

            finger_count = sum(fingers)

            # Jika 2 jari terdeteksi
            if finger_count == 2:
                two_finger_detected = True

    # ==================================================
    # BLUR
    # ==================================================

    if two_finger_detected:

        blurred = cv2.GaussianBlur(
            original,
            (51, 51),
            0
        )

        output = blurred

    else:

        output = original.copy()

    # ==================================================
    # DISPLAY
    # ==================================================

    cv2.imshow("Camera", output)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==================================================
# RELEASE
# ==================================================

cap.release()
hands.close()
cv2.destroyAllWindows()