#!/usr/bin/python

# Import the required modules
import cv2
import os
import sys
import numpy as np
from PIL import Image
# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.createLBPHFaceRecognizer()


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f)
                   for f in os.listdir(path) if not f.endswith('.sad')]
# images will contains face images
    images = []
# labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(image_path)[1].split(
            ".")[0].replace("subject", ""))
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...",
                       image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels

# Path to the Yale Dataset
path = sys.argv[1]
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))
recognizer.save("cont.yaml")
# recognizer.load("/home/arpit/Projects/Survelliance_System/cont.yaml")
