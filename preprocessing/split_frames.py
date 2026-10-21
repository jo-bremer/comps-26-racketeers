import cv2
import os
import glob
from tqdm import tqdm
import sys


def split_frames_from_video(video_path, output_path, frame_size=(512, 288), img_format='jpg'):
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_path, f'{frame_i:04d}.{img_format}')
        cv2.resize(frame, frame_size)
        cv2.imwrite(frame_path, frame)
        frame_i += 1

    cap.release()
    return frame_i

def split_frames_from_videos(videos_path, output_path, frame_size=(512, 288), img_format='jpg'):
    for i, video in tqdm(enumerate(os.scandir(videos_path))):
        video_output_path = os.path.join(output_path, "_"+os.path.basename(video))
        split_frames_from_video(video, video_output_path, frame_size=frame_size)

def main():
    assert len(sys.argv) == 3
    split_frames_from_videos(sys.argv[1], sys.argv[2])

if __name__ == "__main__": main()