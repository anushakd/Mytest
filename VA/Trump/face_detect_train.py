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
count=0
frameCount=0
pframe = [0 for i in range(500)]
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
                    starttime = time.time()
                    #print(starttime)
	            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		    gray = np.array(gray, 'uint8')
		    faces = faceCascade.detectMultiScale(gray)
		    frameCount=frameCount+1
		    for (x, y, w, h) in faces:
			predicted, confidence = recognizer.predict(gray[y: y + h, x: x + w])
			nbr_actual = 1
                        #print("predicted:")
                        #print(predicted)
                        faces1[predicted]=predicted
                        #print(predicted)
                        #print(faces1[predicted])
			if predicted == nbr_actual:
				sub_face = gray[y:y+h, x:x+w]
			
				sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)
		
				gray[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
			else:
				cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)		
		    cv2.imshow('Video', gray)
                    endtime = time.time()
                    #if not (endtime==0) and ((starttime-endtime)>0.05):
                       #endtime=0
                    if ((faces1[1]==1) and (pframe[1]==0)):
                        count=count+1
                        print "####################################################{} is countOne "
                        print(count)
                        #print(starttime-endtime)
                    if frameCount == 40:
                        pframe=faces1
                        frameCount=0
                    print(pframe[1])
                    #print(endtime);
		    if cv2.waitKey(1) & 0xFF == ord('q'):
			break
video_capture.release()
cv2.destroyAllWindows()
