import cv2
import os
import shutil

def extract_frames(source_path, fps, resolution):
    # Load video
    cap = cv2.VideoCapture(source_path)

    # Check if video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS:", fps)
    print("Total frames:", total_frames)

    # Create an empty list to store frames
    frames = []

    # Loop through frames
    frame_index = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if no more frames are available
        if not ret:
            break

        # Process the frame (e.g., save, display, etc.)
        # For example, here we resize the frame and append it to the list
        resized_frame = cv2.resize(frame, (0, 0), fx=resolution, fy=resolution)
        if(frame_index % (int(cap.get(cv2.CAP_PROP_FPS)) // fps) == 0):
            frames.append(resized_frame)  # Append the frame to the list
        frame_index += 1

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Return the list of frames
    return frames


def save_frames(source_path, fps, resolution, dest_path):
    frames = extract_frames(source_path, fps, resolution)
    os.mkdir(dest_path)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(dest_path , "frame%d.jpg" % (i + 1)), frame)

def del_frames(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(e)

# Do something with the frames (e.g., display, save, etc.)
