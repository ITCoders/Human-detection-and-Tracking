import cv2
import pickle
import face_recognition
import numpy as np
from threading import Thread
from imutils.video import WebcamVideoStream
from imutils.video import FPS



class recognize:
    def __init__(self, pickleFilePath, cascadeFilePath):
        self.data = pickle.loads(open(pickleFilePath, "rb").read())
        self.detector = cv2.CascadeClassifier(cascadeFilePath)

    def recognize_face(self, frame):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, flags=cv2.CASCADE_SCALE_IMAGE)
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(self.data["encodings"], encoding)
            name = 'Unknown'
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
                cv2.rectangle(frame, (left, top), (right, bottom),(0, 0, 255), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                if name == 'Unknown':
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
						0.75, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
						0.75, (0, 255, 0), 2)
            
        return frame


"""#camera = cv2.VideoCapture(0)
camera = WebcamVideoStream(src=0).start()
while(True):
    frame1 = camera.read()
    set11 = recognize()
    rsult = set11.recognize_face(frame1)
    cv2.imshow('frame',rsult)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    cv2.imshow('frame',rsult)
camera.stop()
cv2.destroyAllWindows()"""
