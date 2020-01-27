import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(2)

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x) /2), int((p1.y + p2.y)/2)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        #for i in range(36, 48):
        #    x = landmarks.part(i).x
        #    y = landmarks.part(i).y
        #    cv2.circle(frame, (x, y), 1, (0,250,0), 1)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0),1)
        ver_line = cv2.line(frame, center_top, center_bottom, (0,255, 0),1)
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
