import normalization
import detection
import os

for root, dirs, files in os.walk("Dataset\\"):
    for name in files:
        if name.endswith((".mp4", ".MP4")):
            print(os.path.join(root, name))
            path = (os.path.join(root, name)).split("\\")
            n = len(path)
            print(path[n-2])