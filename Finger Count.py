

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    finger = 0

    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label
        handLm = []
        for landmarks in hand_landmarks.landmark:
          handLm.append([landmarks.x, landmarks.y])
        if handLabel == "Left" and handLm[4][0] > handLm[3][0]:
          finger = finger+1
        elif handLabel == "Right" and handLm[4][0] < handLm[3][0]:
          finger = finger+1

        if handLm[8][1] < handLm[6][1]:       
          finger = finger+1
        if handLm[12][1] < handLm[10][1]:     
          finger = finger+1
        if handLm[16][1] < handLm[14][1]:    
          finger = finger+1
        if handLm[20][1] < handLm[18][1]:
          finger = finger+1

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

  
    cv2.putText(image, str(finger), (10, 450), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 10)

    cv2.imshow('img', image)
    if cv2.waitKey(2) & 0xFF == ord("i"):
      break
cap.release()
cv2.destroyAllWindows()