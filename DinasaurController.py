import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, 
                      max_num_hands=2, 
                      min_detection_confidence=0.5, 
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    x = 0
    y = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            hand_type = handedness.classification[0].label
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            
            fingers = 0
            
            if (hand_type == "Right" and landmarks[4].x < landmarks[3].x) or \
               (hand_type == "Left" and landmarks[4].x > landmarks[3].x):
                fingers += 1
            
            for i in range(8, 21, 4):
                if landmarks[i].y < landmarks[i-2].y:
                    fingers += 1
            
            if fingers>0:
                pyautogui.press('space')
                
    cv2.imshow('Finger Counting', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()