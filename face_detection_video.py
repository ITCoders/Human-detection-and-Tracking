import cv2
import sys

cascPath = '/home/unnirajendran/Desktop/face_detect/haarcascade_frontalface_default.xml'
cascPath2 = '/home/unnirajendran/Desktop/face_detect/haarcascade_profileface.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
faceCascade2 = cv2.CascadeClassifier(cascPath2)

#give the name of the input video file

cap = cv2.VideoCapture('video.mp4')

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame =cap.read()
    #convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        flags=0
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    faces2 = faceCascade2.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        flags=0
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
        

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()