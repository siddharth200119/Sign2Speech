import cv2
import os

# Load video
cap = cv2.VideoCapture('D:\\ML Project\\ISL_CSLRT_Corpus\\Videos_Sentence_Level\\are you hiding something\\MVI_6209.MP4')

# Check if video file was successfully opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get frames per second (fps) and total number of frames in the video
fps = 10
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS:", fps)
print("Total frames:", total_frames)

# Specify the directory to save the extracted frames
output_directory = 'D:\\ML Project\\ISL_CSLRT_Corpus\\Final_Images'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through frames
frame_index = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if no more frames are available
    if not ret:
        break

    # Process the frame (e.g., save, display, etc.)
    # For example, here we resize the frame to half its original size and save it as an image file
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame_filename = os.path.join(output_directory, f"frame_{frame_index:04d}.jpg")  # Save frames with index as filename
    if(frame_index % 3 == 0):
        cv2.imwrite(frame_filename, resized_frame)  # Save the frame as an image file
        print(f"Saved frame {frame_index} as {frame_filename}")
    frame_index += 1

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()