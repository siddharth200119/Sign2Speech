import cv2
import os

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
            print(f"Extracted frame {frame_index}")
        frame_index += 1

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Return the list of frames
    return frames

# User inputs
# source_path = input("Enter source path of video file: ")
# fps = int(input("Enter desired FPS for extracted frames: "))
# resolution = float(input("Enter desired resolution for extracted frames (e.g., 0.5 for half resolution): "))

# Call the function with user inputs
frames = extract_frames("C:\\Users\\JAY\Downloads\\are you free today (2).mp4", 10, 0.5)
print(frames)

# Do something with the frames (e.g., display, save, etc.)
