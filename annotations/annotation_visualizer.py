import cv2
import numpy as np
from config import *
import annotations.read_annotations as read_annotations

def visualize_ball_positions(ball_position_annotations_path : str, video_input_path : str, video_output_path : str):
    # Creating a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_input_path)
    video_output = cv2.VideoWriter(
        video_output_path, cv2.VideoWriter_fourcc(*'XVID'), VIDEO_FPS, VIDEO_SIZE)
    frames = read_annotations.read_ball_position_annotations(ball_position_annotations_path)
    
    print()
    frame_n = 0

    # Loop until the end of the video
    while (cap.isOpened()):
        
        print(f"FRAME {frame_n}", end="\r")
        

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_n in frames:
            cv2.circle(frame, center=frames[frame_n], radius=20, color=(255,255,255), thickness=5)
        
        video_output.write(frame)
        frame_n += 1

    video_output.release()
    cap.release()