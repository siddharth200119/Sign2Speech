import cv2
from os import listdir
import mediapipe as mp
import numpy as np
import math

mp_holistic = mp.solutions.holistic 

def detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    points = model.process(image)
    return points

def angle_extraction(points):
    angles = []
    for point in range(len(points) - 2):
        v1 = points[point] - points[point + 1]
        v2 = points[point + 2] - points[point + 1]
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(cosine_angle))
        if(math.isnan(angle)):
            angles.append(0)
        else:
            angles.append(angle)
    return np.array(angles)

def video_detection(location):
    cap = cv2.VideoCapture(location)
    result = {
        "pose_points": [],
        "face_points": [],
        "lh_points": [],
        "rh_points": [],
        "pose_angles": [],
        "face_angles": [],
        "lh_angles": [],
        "rh_angles": []
    }
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            points = detection(frame, holistic)

            pose = np.array([[res.x, res.y, res.z] for res in points.pose_landmarks.landmark]) if points.pose_landmarks else np.zeros((33,3))
            face = np.array([[res.x, res.y, res.z] for res in points.face_landmarks.landmark]) if points.face_landmarks else np.zeros((468,3))
            lh = np.array([[res.x, res.y, res.z] for res in points.left_hand_landmarks.landmark]) if points.left_hand_landmarks else np.zeros((21,3))
            rh = np.array([[res.x, res.y, res.z] for res in points.right_hand_landmarks.landmark]) if points.right_hand_landmarks else np.zeros((21,3))

            result["pose_points"].append(pose)
            result["face_points"].append(face)
            result["lh_points"].append(lh)
            result["rh_points"].append(rh)

            result["pose_angles"].append(angle_extraction(pose))
            result["face_angles"].append(angle_extraction(face))
            result["lh_angles"].append(angle_extraction(lh))            
            result["rh_angles"].append(angle_extraction(rh))
            
        cap.release()
        cv2.destroyAllWindows()
        return result

def frame_detection(location):
    frame = cv2.imread(location, cv2.IMREAD_UNCHANGED)
    result = {
        "pose_points": [],
        "face_points": [],
        "lh_points": [],
        "rh_points": [],
        "pose_angles": [],
        "face_angles": [],
        "lh_angles": [],
        "rh_angles": []
    }
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        points = detection(frame, holistic)

        pose = np.array([[res.x, res.y, res.z] for res in points.pose_landmarks.landmark]) if points.pose_landmarks else np.zeros((33,3))
        face = np.array([[res.x, res.y, res.z] for res in points.face_landmarks.landmark]) if points.face_landmarks else np.zeros((468,3))
        lh = np.array([[res.x, res.y, res.z] for res in points.left_hand_landmarks.landmark]) if points.left_hand_landmarks else np.zeros((21,3))
        rh = np.array([[res.x, res.y, res.z] for res in points.right_hand_landmarks.landmark]) if points.right_hand_landmarks else np.zeros((21,3))

        result["pose_points"].append(pose)
        result["face_points"].append(face)
        result["lh_points"].append(lh)
        result["rh_points"].append(rh)

        result["pose_angles"].append(angle_extraction(pose))
        result["face_angles"].append(angle_extraction(face))
        result["lh_angles"].append(angle_extraction(lh))            
        result["rh_angles"].append(angle_extraction(rh))
            
    cv2.destroyAllWindows()
    return result

def returned_frames_detection(frames):
    result = {
        "pose_points": [],
        "face_points": [],
        "lh_points": [],
        "rh_points": [],
        "pose_angles": [],
        "face_angles": [],
        "lh_angles": [],
        "rh_angles": []
    }
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        for frame in frames:
            points = detection(frame, holistic)

            pose = np.array([[res.x, res.y, res.z] for res in points.pose_landmarks.landmark]) if points.pose_landmarks else np.zeros((33,3))
            face = np.array([[res.x, res.y, res.z] for res in points.face_landmarks.landmark]) if points.face_landmarks else np.zeros((468,3))
            lh = np.array([[res.x, res.y, res.z] for res in points.left_hand_landmarks.landmark]) if points.left_hand_landmarks else np.zeros((21,3))
            rh = np.array([[res.x, res.y, res.z] for res in points.right_hand_landmarks.landmark]) if points.right_hand_landmarks else np.zeros((21,3))

            result["pose_points"].append(pose)
            result["face_points"].append(face)
            result["lh_points"].append(lh)
            result["rh_points"].append(rh)

            result["pose_angles"].append(angle_extraction(pose))
            result["face_angles"].append(angle_extraction(face))
            result["lh_angles"].append(angle_extraction(lh))            
            result["rh_angles"].append(angle_extraction(rh))

    return result
