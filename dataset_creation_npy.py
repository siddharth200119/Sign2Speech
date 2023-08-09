import normalization
import detection
import os
import numpy as np
import csv
import pandas

def create_entry(file_path, entry_name, fps, scale):

    frames = normalization.extract_frames(file_path, fps, scale)
    results = detection.returned_frames_detection(frames)
    results["gesture"] = entry_name

    return results

def create_dataset(dataset_path):

    dataset = {
        "gesture": [],
        "pose_points": [],
        "face_points": [],
        "lh_points": [],
        "rh_points": [],
        "pose_angles": [],
        "face_angles": [],
        "lh_angles": [],
        "rh_angles": []
    }

    fps = float(input("enter fps: "))
    scale = float(input("enter scale: "))

    for root, dirs, files in os.walk(dataset_path + "\\"):
        for name in files:
            if name.endswith((".mp4", ".MP4")):
                path = os.path.join(root, name)
                path_split = (os.path.join(root, name)).split("\\")
                entry_name = path_split[len(path_split)-2]
                print(path)
                results = create_entry(path, entry_name, fps, scale)
                for key in dataset:
                    dataset[key].append(results[key])
                
    for key in dataset:
        np.save("Dataset_npy\\" + key, np.array(dataset[key]))

create_dataset("Dataset_test")