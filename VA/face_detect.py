#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import time

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.createLBPHFaceRecognizer()


def get_training_data(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) ]
    images = []
    # labels will be the image name
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        label = int(os.path.split(image_path)[1].split(".")[0])
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(label)
    return images, labels

# Path to the My Dataset
path = './my'
images, labels = get_training_data(path)
cv2.destroyAllWindows()
x = input("Enter a choice: 1 for detectig faces and 2 for bluring my face ")
# Perform the tranining.
recognizer.train(images, np.array(labels))

video_capture = cv2.VideoCapture(0)

predicted=0
endtime=0
starttime=0
count1=0
count2=0
count3=0
frameCount=0
currtime=0
flag=0
prevFace = [0 for i in range(500)]
if x==1:
	while True:
	    ret, frame = video_capture.read()
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    gray = np.array(gray, 'uint8')
	    faces = faceCascade.detectMultiScale(gray)
	   
	    for (x, y, w, h) in faces:
		predicted, confidence = recognizer.predict(gray[y: y + h, x: x + w])
		cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
	    cv2.imshow('Video', gray)

	    if cv2.waitKey(1) & 0xFF == ord('q'):
		break
else:
	while True:
		    ret, frame = video_capture.read()
                    faces1=[0 for i in range(102)]
                    value=1
                    starttime = time.time()
	            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		    gray = np.array(gray, 'uint8')
		    faces = faceCascade.detectMultiScale(gray)
		    for (x, y, w, h) in faces:
			predicted, confidence = recognizer.predict(gray[y: y + h, x: x + w])
			nbr_actual1 = 1
                        nbr_actual2 = 2
                        nbr_actual3 = 3
			if(confidence < 40):
				print "Label {} is correctlly recognized with confidence of {}".format(predicted, confidence)
				faces1[predicted]=predicted
				if predicted == nbr_actual1:
					sub_face = gray[y:y+h, x:x+w]			
					sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)		
					gray[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
		                if predicted == nbr_actual2:
					sub_face = gray[y:y+h, x:x+w]			
					sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)
					gray[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
		                if predicted == nbr_actual3:
					sub_face = gray[y:y+h, x:x+w]		
					sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)	
					gray[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face		
		    cv2.imshow('Video', gray)
                    endtime = time.time()
                    diff = (endtime-starttime)
                    currtime=currtime+diff
                    if ((faces1[1]==1) and (prevFace[1]==0)):
                        count1=count1+1
                        print "Label 1 Count is : "
                        print(count1)
                        flag=0
                    if ((faces1[2]==2) and (prevFace[2]==0)):
                        count2=count2+1
                        print "Label 2 count is : "
                        print(count2)
                        flag=0
                    if ((faces1[3]==3) and (prevFace[3]==0)):
                        count3=count3+1
                        print "Label 3 Count is : "
                        print(count3)
                        flag=0
                    if flag == 0 or currtime > 2.0000000000:
                        prevFace=faces1  
                        currtime=0
                        flag=1
		    if cv2.waitKey(1) & 0xFF == ord('q'):
			break
video_capture.release()
cv2.destroyAllWindows()
