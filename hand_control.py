import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize previous x-coordinate of wrist for movement tracking
prev_wrist_x = None

# Start MediaPipe Hand Tracking
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates for Index Finger, Thumb, and Wrist
                index_tip = hand_landmarks.landmark[8]  # Index Finger Tip
                index_base = hand_landmarks.landmark[6]  # Index Finger Base
                thumb_tip = hand_landmarks.landmark[4]  # Thumb Tip
                thumb_base = hand_landmarks.landmark[2]  # Thumb Base
                wrist = hand_landmarks.landmark[0]  # Wrist position

                # Detect Jump (Up Arrow) using Index Finger
                if index_tip.y < index_base.y:  # Index Finger Up (Jump)
                    pyautogui.press('up')

                # Detect Slide (Down Arrow) using Thumbs Up
                if thumb_tip.y < thumb_base.y:  # Thumbs Up (Slide)
                    pyautogui.press('down')

                # Detect Left and Right Movement using Palm (Wrist)
                if prev_wrist_x is not None:
                    if wrist.x < prev_wrist_x - 0.02:  # Move Left
                        pyautogui.press('left')
                    elif wrist.x > prev_wrist_x + 0.02:  # Move Right
                        pyautogui.press('right')

                # Update previous wrist position for next frame comparison
                prev_wrist_x = wrist.x

        cv2.imshow("Hand Control Game", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
