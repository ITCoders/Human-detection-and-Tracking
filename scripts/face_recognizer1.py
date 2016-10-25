#!/usr/bin/python

# Import the required modules
import cv2
import os
import numpy as np
from PIL import Image
# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.face.createLBPHFaceRecognizer()


def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f)
                   for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
    # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
    # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(
            ".")[0].replace("subject", ""))
    # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
    # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...",
                       image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

# Path to the Yale Dataset
path = './yalefaces'
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

# Append the images with the extension .sad into image_paths
image_paths = [os.path.join(path, f)
               for f in os.listdir(path) if f.endswith('.sad')]
for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_actual = int(os.path.split(image_path)[
                         1].split(".")[0].replace("subject", ""))
        # print(nbr_predicted)
        label_predicted, confidence_pred = nbr_predicted
        # print(confidence_pred)
        if nbr_actual == label_predicted:
            print("{} is Correctly Recognized with confidence {}".format(
                nbr_actual, confidence_pred))
        else:
            print("{} is Incorrect Recognized as {} with confidence {}".format(
                nbr_actual, label_predicted, confidence_pred))
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)
