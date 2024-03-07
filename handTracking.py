
import cv2
import time
import mediapipe as mp

cap=cv2.VideoCapture(0)
pTime=0
cTime=0

mpHand=mp.solutions.hands
hands=mpHand.Hands(max_num_hands=2)
mpDraw=mp.solutions.drawing_utils

while True:
    success, img =cap.read()
    imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    print(result.multi_hand_landmarks)
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
            for id,lm in enumerate(handLms.landmark):
               # print(id,lm)
               h,w,c=img.shape
               cx,cy=int(lm.x*w),int(lm.y*h)
               
               if id==0:
                   cv2.circle(img, (cx,cy),10,(186,85,211),cv2.FILLED)
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,"FPS:"+str(int(fps)),(10,75),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,0),3)    
    cv2.imshow("img",img)
    if cv2.waitKey(1) & 0xFF ==ord('i'):
        break
cap.release()
cv2.destroyAllWindows()