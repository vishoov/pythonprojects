import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap=cv2.VideoCapture(0)
fd=FaceDetector()

while True:
    ret,video=cap.read()
    img,face=fd.findFaces(video)
    if face:
        x,y,w,h=face[0]["bbox"]
        print(x,y,w,h)
        
        faceimg=video[y:y+h,x:x+w]
        faceimg=cv2.blur(faceimg,(40,40),1)
        video[y:y+h,x:x+w]=faceimg
    
    
    cv2.imshow("video",video)
    cv2.waitKey(1)