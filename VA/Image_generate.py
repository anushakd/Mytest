#Author:Anusha Kowdeed.
import cv2
import sys
import numpy
import os
from os import rename, listdir

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
video_capture = cv2.VideoCapture(0)
sem=0
id=input("Enter No Of images To be generated")
while True:
   
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,1.3,5
    )

   
    for (x, y, w, h) in faces:
	cv2.imwrite("./my/"+"1_"+str(sem)+".jpeg",gray[y:y+h,x:x+w])
	print "{} is sem".format(sem)
        sem = sem+1
	cv2.waitKey(100)

    
    cv2.imshow('Video', frame)
 
    if sem>=int(id):
        break
    if cv2.waitKey(1) & 0xFF == ord('q') | sem==100:
        break
video_capture.release()
for filename in os.listdir("./my/"):
	if filename.startswith("1_"):
		target= "./my/1.jpeg"+ filename.split(".")[0][2:]
		print "{} is target".format(target)
		os.rename("./my/"+filename,target)
cv2.destroyAllWindows()
