import normalization
import detection
import os
import numpy as np
import csv
import pandas

header_written = False

def create_entry(path, entry_name, fps, scale):
    frames = normalization.extract_frames(path, fps, scale)
    results = detection.returned_frames_detection(frames)
    results["gesture"] = entry_name
    fields = ["gesture", "pose_points", "face_points", "lh_points", "rh_points", "pose_angles", "face_angles", "lh_angles", "rh_angles"]
    output_file = "output" + "fps" + str(fps) + "scale" + str(scale) + ".csv"
    global header_written
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fields)
        if(header_written == False):
            writer.writeheader()
            header_written = True
        writer.writerows([results])

fps = float(input("enter fps: "))
scale = float(input("enter scale: "))


for root, dirs, files in os.walk("Dataset\\"):
    for name in files:
        if name.endswith((".mp4", ".MP4")):
            path = os.path.join(root, name)
            path_split = (os.path.join(root, name)).split("\\")
            entry_name = path_split[len(path_split)-2]
            create_entry(path, entry_name, fps, scale)
            print(path)