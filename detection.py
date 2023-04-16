import cv2
import os
from os import listdir
import mediapipe as mp
import numpy as np

#to detect points

mp_holistic = mp.solutions.holistic 

def detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    points = model.process(image)
    return points

def video_detection(location):
    cap = cv2.VideoCapture(location)
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            points = detection(frame, holistic)
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in points.pose_landmarks.landmark]).flatten() if points.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in points.face_landmarks.landmark]).flatten() if points.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in points.left_hand_landmarks.landmark]).flatten() if points.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in points.right_hand_landmarks.landmark]).flatten() if points.right_hand_landmarks else np.zeros(21*3)
            result = np.concatenate([pose, face, lh, rh])
        cap.release()
        cv2.destroyAllWindows()
        return result