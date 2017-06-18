#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image

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
path1= './att'
x = input("Enter a choice: 1 for att database and 2 for my database ")
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
path2 = path1 if x==1 else path
#print "{} is path ".format(path2)
images, labels = get_training_data(path2)
cv2.destroyAllWindows()

# Perform the tranining and test on one sample image.
imgpath = './my/1.detect' if x==2 else './att/8.pgm2'
#print "{} is path ".format(imgpath)
recognizer.train(images, np.array(labels))
predict_image_pil = Image.open(imgpath).convert('L')
predict_image = np.array(predict_image_pil, 'uint8')
faces = faceCascade.detectMultiScale(predict_image)
for (x, y, w, h) in faces:
    nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
    nbr_actual = int(imgpath.split('/')[2].split('.')[0])
    if nbr_actual == nbr_predicted:
            print("{} is Correctly Recognized ".format(nbr_predicted))
    else:
            print("{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted))

video_capture = cv2.VideoCapture(0)

predicted=0
max_threshold=41
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, 'uint8')
    faces = faceCascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        predicted, confidence = recognizer.predict(gray[y: y + h, x: x + w])
        if confidence<max_threshold:
         returnVal=1
         print ("{} is Correctly Recognized with confidence {}".format(predicted, confidence))
        if predicted == returnVal:
            sub_face = gray[y:y+h, x:x+w]     
            sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)    
            gray[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
        else:
            cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Video', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
