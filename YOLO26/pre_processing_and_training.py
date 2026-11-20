from pathlib import Path
from ultralytics import YOLO
import csv
import json
import math
import itertools
import ast
import cv2
import os
import glob
import shutil
import pandas as pd




PROJECT_DIR = Path(__file__).resolve().parent

'''
This function obtains lists for the videos in the RacketVision-defined training, validation, and 
testing splits.
'''
def load_video_splits():
    training_videos = []
    with open('train.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        for match in data:
            training_videos.append(match[0])
            
    val_videos = []
    with open('val.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        for match in data:
            val_videos.append(match[0])
            
    test_videos = []
    with open('test.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        for match in data:
            test_videos.append(match[0])

    return training_videos, val_videos, test_videos

'''
Gets the ground truth .csv files from RacketVision into the labels folder in the YOLO26 directory.
'''
def copy_ground_truth_csvs(label_dir, match_numbers, label_folder):
    moved_count = 0 # This will be equal to 431 after the loop below runs.
    for match in match_numbers:
        match = "match" + str(match)
        print("Match:", match)
        ground_truth_dir = label_dir / match / "csv"
        print("Ground Truth Directory:", ground_truth_dir)
        match_name = match + ".csv"
        label_name = label_folder + "/" + match_name
        csv_file = glob.glob(os.path.join(ground_truth_dir, "*.csv"))
        print(csv_file)
        full_path = os.path.join(label_folder, match_name)
        file_name = os.path.basename(csv_file[0])
        destination_path = os.path.join(label_folder, file_name)
        
        # Move the file.
        shutil.copy2(csv_file[0], full_path)
        print(f"Moved: {file_name} -> {label_folder}/")
        moved_count += 1
     
    
    print(f"\nSuccess! Successfully moved {moved_count} CSV files.")

    return moved_count

'''
    The function creates .jpg files for each RacketVision annotated frame from the videos (
    through line 156). This consists of using the VideoCapture() function from from the cv2
    package to create an object that captures the video ("cap"), then having a while loop,
    "while cap.isOpened:." Then, the cap.read() function is used to extract a frame for each
    iteration of the while loop and as long as the current frame number is one of the frame
    numbers in the .csv of ground truth annotated frames for the current video, the .jpg for the
    current frame is written to the appropriate test, train, or val subdirectory in the images
    folder in the YOLO26 directory (based on which split RacketVision assigned the current video 
    to).
    '''
def extract_annotated_frames(training_videos, val_videos, test_videos):
    # Set up directories pointing to the actual directory with the videos from the dataset.
    video_dir = Path.home() / "data/RacketVision/tennis/videos"
    train_output_folder = PROJECT_DIR / "images" / "train"
    val_output_folder = PROJECT_DIR / "images" / "val"
    test_output_folder = PROJECT_DIR / "images" / "test"
    
    
    # Find all .mp4 videos in the video directory.
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        print(f"No .mp4 videos found in {video_dir}")
        return

    frame_count = 0

    # Loop through each video file found.
    for video_path in video_files:
        frame_number = 0
        print(f"Processing video: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path.name}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  # Fallback if FPS reading fails.
            fps = 30
            
        video_frame_idx = 0

        while cap.isOpened():
            
            # Read every single frame sequentially.
            ret, frame = cap.read()
            if not ret:
                break

            target_frame = video_frame_idx
            
            # Gets just the match names from the video name.
            match_name = video_path.stem.split("_")[0] 
            
            
            if not ret:
                break  # End of the current video.
                
            # Use unique filenames so videos do not overwrite the frames of other videos.
            if match_name in training_videos:
                img_name = train_output_folder / f"{match_name}_frame_{frame_number}.jpg"
            
            
            if match_name in val_videos:
                img_name = val_output_folder / f"{match_name}_frame_{frame_number}.jpg"
            
            
            if match_name in test_videos:
                img_name = test_output_folder / f"{match_name}_frame_{frame_number}.jpg"
            
            label_file = match_name + ".csv"
            
            match_df = pd.read_csv(PROJECT_DIR / "labels" / label_file)
            
            if frame_number in match_df["Frame"].values:
                cv2.imwrite(str(img_name), frame)
            
            video_frame_idx += 1
            frame_count += 1
            frame_number += 1

        cap.release()
        
    print(f"Successfully extracted {frame_count} frames.")

    return frame_count

'''
This file creates YOLO labels for a split of the dataset consisting of one .txt file for each
frame .jpg file that has been created for the split input into the function. These .txt files
consist of one line assuming that a ball is visible in the current frame and are otherwise empty.
If there is a ball visible in the current frame, the one line of the .txt file looks like this:

                    class_id    center_x    center_y    box_width   box_height

class_id is always 0 because this is the id of the tennis_ball class as defined in videos.yaml.
center_x is the tennis ball's ground truth horizontal position in the frame relative to the size
of the frame (calculated as: annotated ball x-coordinate/image width). For instance, a value of 0 
is at the left edge of the frame, a value of 0.5 is halfway across the frame, and a value of 1 is 
at the right edge of the frame. center_y is the tennis ball's ground truth vertical position in 
the frame relative to the size of the frame (calculated as: annotated ball y-coordinate/image 
height). For instance, a value of 0 is at the bottom of the frame, a value of 0.5 halfway up the 
frame, and a value of 1 is at the top of the frame. box_width and box_height plus center_x and 
center_y all combine to form 20 pixel by 20 pixel ground truth bounding boxes around the 
RacketVision ground truth annotated ball locations. 

The YOLO labels are created by looping through all of the newly created .jpg files for the input
split and then extracting the ball-visibility and location information from the corresponding
frame row for the particular current video .csv from RacketVision's ground truth annotations and 
using that to obtain the values in the .txt YOLO labels.
'''
def create_yolo_labels(split):
    split_path = Path(f"./images/{split}")
    
    file_counter = 0
    # Loop through all of the frame .jpg files for the input split.
    for jpg_file in split_path.glob("*.jpg"):
        file_name = jpg_file.stem + ".txt"
        file_path = PROJECT_DIR / "labels" / split / file_name
        
        print(f"Creating {file_name}")
        # Get values to put in the .txt files from the .csv label files.
        match_number = jpg_file.stem.split("_")[0]
        frame_number = int(jpg_file.stem.split("_")[2])
        
        label_dir = Path("./labels")
        csv_name = f"{match_number}.csv"
        csv_path = label_dir / csv_name
        match_csv = pd.read_csv(csv_path)
        
        matching_row = match_csv.loc[match_csv["Frame"] == frame_number]
        
        csv_frame_row = matching_row.iloc[0]
        
        
        if csv_frame_row["Visibility"] == 1: # If this is not true, the .txt file will be empty.
            class_id = 0
        
            image = cv2.imread(jpg_file)
            image_height, image_width = image.shape[:2]
            
            x_center = csv_frame_row["X"]/image_width
            y_center = csv_frame_row["Y"]/image_height 
            
            # Creating 20 pixel by 20 pixel annotated bounding boxes.
            box_width = 20 / image_width  
            box_height = 20 / image_height
        
            with open(file_path, "w") as file:
                file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
        
        else:
             with open(file_path, "w") as file:
                file_path.write_text("", encoding="utf-8")
        
        file_counter += 1

    print(f"Created {file_counter} .txt files for {split}.")


def main():
    training_videos, val_videos, test_videos = load_video_splits()

    # Possibly make the path below more universal
    label_dir = Path.home() / "data/RacketVision/tennis/all"
    
    match_numbers = list(range(1, 432))
   
    label_folder = "./labels"
    

    # Uncomment this call to get the ground truth .csv files from RacketVision into this directory
    # if you have not yet done so.
    # copy_ground_truth_csvs(label_dir, match_numbers, label_folder)


    frame_count = extract_annotated_frames(training_videos, val_videos, test_videos)
    if frame_count is None: 
        return # Stops main if no video files were found.

    '''
    Create ground truth YOLO labels with bounding box information to use as input for training,
    validation, and testing.
    '''
    create_yolo_labels("test")
    create_yolo_labels("train")
    create_yolo_labels("val")


    model = YOLO(str(PROJECT_DIR / "yolo26x.pt"))
    
    '''
    Train the pretrained YOLO26x model. The results of training will be put into this directory:
    ".../YOLO26/runs/tennis/yolo26_run1".
    '''
    results = model.train(
        data = str(PROJECT_DIR / "videos.yaml"),
        epochs = 30,
        patience = 10,
        imgsz = 1280,
        batch = -1,
        device = 0,
        optimizer = "AdamW",
        lr0 = 0.001,
        warmup_bias_lr = 0.0,
        seed = 0,
        project = str(PROJECT_DIR / "runs" / "tennis"),
        name = "yolo26_run1",
    )


if __name__ == "__main__":
    main()
