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

def main():
    # Get lists for the videos in the RacketVision-defined training, validation, and testing splits.
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
    
    
    
    # Possibly make the path below more universal
    label_dir = Path.home() / "data/RacketVision/tennis/all"
    
    match_numbers = list(range(1, 432))
   
    label_folder = "./labels"
    
    moved_count = 0 # Should equal 431 after the loop below runs.
   
   
   # Uncomment this code to get the ground truth .csv files from RacketVision into this folder.
    '''
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
        
        # Move the file
        shutil.copy2(csv_file[0], full_path)
        print(f"Moved: {file_name} -> {label_folder}/")
        moved_count += 1
     
    
    print(f"\nSuccess! Successfully moved {moved_count} CSV files.")
    '''
    
    video_count_multibox = 0
    
    
    
    
    '''
    The lines below create .jpg files for each RacketVision annotated frame from the videos (
    through line 156). This consists of using the VideoCapture() function from from the cv2
    package to create an object that captures the video ("cap"), then having a while loop,
    "while cap.isOpened:." Then, the cap.read() function is used to extract a frame for each
    iteration of the while loop and as long as the current frame number is one of the frame
    numbers in the .csv of ground truth annotated frames for the current video, the .jpg for the
    current frame is written to the appropriate test, train, or val subdirectory in the images
    folder in the YOLO26 directory (based on which split RacketVision assigned the current video 
    to).
    '''
    # Set up directories pointing to the actual directory with the videos from the dataset.
    video_dir = Path.home() / "data/RacketVision/tennis/videos"
    train_output_folder = PROJECT_DIR / "images" / "train"
    val_output_folder = PROJECT_DIR / "images" / "val"
    test_output_folder = PROJECT_DIR / "images" / "test"
    #output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all .mp4 videos in the directory.
    
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
                
            # Use unique filenames so videos do not overwrite each other's frames.
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
    
    
    test_path = Path("./images/test")
    
    file_counter = 0
    for jpg_file in test_path.glob("*.jpg"):
        file_name = jpg_file.stem + ".txt"
        #file_path = test_output_folder / file_name
        file_path = PROJECT_DIR / "labels" / "test" / file_name
        
        print(f"Creating {file_name}")
        # Get values to put in the .txt files from the .csv label files
        match_number = jpg_file.stem.split("_")[0]
        frame_number = int(jpg_file.stem.split("_")[2])
        
        label_dir = Path("./labels")
        csv_name = f"{match_number}.csv"
        csv_path = label_dir / csv_name
        match_csv = pd.read_csv(csv_path)
        
        matching_row = match_csv.loc[match_csv["Frame"] == frame_number]
        
        csv_frame_row = matching_row.iloc[0]
        #csv_frame_row = match_csv.loc[match_csv["Frame"] == frame_number][0]
        
        if csv_frame_row["Visibility"] == 1: # If this is not true, the .txt file will be empty
            class_id = 0
        
            image = cv2.imread(jpg_file)
            image_height, image_width = image.shape[:2]
            
            x_center = csv_frame_row["X"]/image_width
            y_center = csv_frame_row["Y"]/image_height 
            
            # Below: assuming 20x20 annotated bounding boxes -possibly change or do cross-validation on this
            box_width = 20 / image_width  
            box_height = 20 / image_height
        
            with open(file_path, "w") as file:
                file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
        
        else:
             with open(file_path, "w") as file:
                file_path.write_text("", encoding="utf-8")
        
        file_counter += 1
        
    train_path = Path("./images/train")
    for jpg_file in train_path.glob("*.jpg"):
        file_name = jpg_file.stem + ".txt"
        #file_path = train_output_folder / file_name
        file_path = PROJECT_DIR / "labels" / "train" / file_name
        
        print(f"Creating {file_name}")
        # Get values to put in the .txt files from the .csv label files
        match_number = jpg_file.stem.split("_")[0]
        frame_number = int(jpg_file.stem.split("_")[2])
        
        label_dir = Path("./labels")
        csv_name = f"{match_number}.csv"
        csv_path = label_dir / csv_name
        match_csv = pd.read_csv(csv_path)
        
        matching_row = match_csv.loc[match_csv["Frame"] == frame_number]
        
        csv_frame_row = matching_row.iloc[0]
        #csv_frame_row = match_csv.loc[match_csv["Frame"] == frame_number][0]
        
        if csv_frame_row["Visibility"] == 1: # If this is not true, the .txt file will be empty
            class_id = 0
        
            image = cv2.imread(jpg_file)
            image_height, image_width = image.shape[:2]
            
            x_center = csv_frame_row["X"]/image_width
            y_center = csv_frame_row["Y"]/image_height 
            
            # Below: assuming 20x20 annotated bounding boxes -possibly change or do cross-validation on this
            box_width = 20 / image_width  
            box_height = 20 / image_height
        
            with open(file_path, "w") as file:
                file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
        
        else:
             with open(file_path, "w") as file:
                file_path.write_text("", encoding="utf-8")
        
        file_counter += 1
        
    val_path = Path("./images/val")
    for jpg_file in val_path.glob("*.jpg"):
        file_name = jpg_file.stem + ".txt"
        #file_path = val_output_folder / file_name
        file_path = PROJECT_DIR / "labels" / "val" / file_name
        
        print(f"Creating {file_name}")
        # Get values to put in the .txt files from the .csv label files
        match_number = jpg_file.stem.split("_")[0]
        frame_number = int(jpg_file.stem.split("_")[2])
        
        label_dir = Path("./labels")
        csv_name = f"{match_number}.csv"
        csv_path = label_dir / csv_name
        match_csv = pd.read_csv(csv_path)
        
        matching_row = match_csv.loc[match_csv["Frame"] == frame_number]
        
        csv_frame_row = matching_row.iloc[0]
        #csv_frame_row = match_csv.loc[match_csv["Frame"] == frame_number][0]
        
        if csv_frame_row["Visibility"] == 1: # If this is not true, the .txt file will be empty
            class_id = 0
        
            image = cv2.imread(jpg_file)
            image_height, image_width = image.shape[:2]
            
            x_center = csv_frame_row["X"]/image_width
            y_center = csv_frame_row["Y"]/image_height 
            
            # Below: assuming 20x20 annotated bounding boxes -possibly change or do cross-validation on this
            box_width = 20 / image_width  
            box_height = 20 / image_height
        
            with open(file_path, "w") as file:
                file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
        
        else:
             with open(file_path, "w") as file:
                file_path.write_text("", encoding="utf-8")
        
        file_counter += 1
        
    print(f"Created {file_counter} .txt files.")
    
    model = YOLO(str(PROJECT_DIR / "yolo26x.pt"))
    #results = model.train(data = "videos.yaml", epochs = 50, imgsz = 1280)
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
    
    #model = YOLO(str(PROJECT_DIR / "yolo26x.pt"))
    '''
    #possibly change imgsz-not exactly sure what that means right now
    #results = model.train(data = "videos.yaml", epochs = 50, imgsz = 1280)
    
    # write code to get the frames from just the training videos into a training folder, get the
    # frames from just the validation videos into a validation folder, and get the frames from
    # just the testing videos into a testing folder. Look at the JSON files in the data to get
    # the split for this.
     
    '''
    '''
    for video in sorted(video_dir.glob("*.mp4")):
        results = model.predict(
            source=str(video),
            save=False, # save a new video
            conf=0.2,
            classes=[32],
            device=0,
            stream=True,
            project=str(PROJECT_DIR / "runs/detect"),
        )
        # write a new csv to include frame number, visibility, box dimensions, confidence
        csv_direct = PROJECT_DIR / "output_csv"
        output_csv = csv_direct / f"{video.stem}.csv"
        with open(output_csv, "w+", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_number",
                "visibility",
                "xyxy",
                "confidence",
            ])
            frame_count = 0
            noDetect_count = 0
            box_count = 0
            detection_count = 0
            max_box_count = 0
            frame_number = 0 #assuming we start from frame 1
            for result in results:
                frame_count += 1
                frame_number = frame_count
                boxes = result.boxes
                if len(boxes) > max_box_count:
                    max_box_count=len(boxes)
                if len(boxes) == 0:
                    writer.writerow([
                                frame_number,
                                0,
                                "",
                                "",
                            ])
                    noDetect_count += 1
                    continue
                boxesPerFrame = []
                confPerFrame = []
                for box in boxes:
                    box_count +=1 
                    xyxy=box.xyxy[0].tolist() 
                    confidence = box.conf.item()
                    boxesPerFrame.append(xyxy) #[[x, y, x, y], [x, y, x, y], [x, y, x, y]]
                    confPerFrame.append(confidence) #[x, y, z, ...]
                
                writer.writerow([
                                frame_number,
                                1,
                                boxesPerFrame, 
                                confPerFrame])
            detection_count = frame_count - noDetect_count
            writer.writerow(["Detection count is", detection_count, "the ratio is",detection_count/frame_count, "total frames", frame_count
                             ])
            if max_box_count > 1:
                video_count_multibox += 1    
            print("the Detection count is ", detection_count, "the ratio is", detection_count/frame_count)
            print(f"Completed inference on {frame_count} frames.")
     
            f.flush()
            f.seek(0)
            csv_reader = list(csv.reader(f))
            center_points = []
            i = 0
            frame_number = 1
          
            for row in itertools.islice(csv_reader, 1, 11): # Loop through first 10 .csv rows
                coords = []
                frame_center_points = []
                if len(row[2]) > 0:
                    for xy in row[2]:
                        coords.append(xy)
                    
                    cur_group = ""
                    new_coords = []
                    for coord in coords:
                        if coord != "," and coord != "[" and coord != "]":
                            cur_group += coord
                        else:
                            new_coords.append(cur_group)
                            cur_group = ""

                    filtered_coords = [x for x in new_coords if x != "" and x != " "]
                    filtered_coords = list(map(float, filtered_coords))
                    curr_ball_coords = []
                    final_filtered_coords = [] 
                    print(filtered_coords)       
                    for coord in filtered_coords:
                    
                        curr_ball_coords.append(coord)
                        print(curr_ball_coords)

                        if len(curr_ball_coords) == 4:
                            print("here")
                            final_filtered_coords.append(curr_ball_coords)
                            curr_ball_coords = []

                    #print(final_filtered_coords)
                    for coord in final_filtered_coords:
                        frame_center_points.append([0.5*(coord[0] + coord[2]), 0.5*(coord[1] + coord[3])]) 
                    
                    center_points.append(frame_center_points)
                    frame_number += 1
            
            stationary_points = []
            for i in range(1, len(center_points)):
                previous_frame = center_points[i-1]
                current_frame = center_points[i]
                for current_point in current_frame:
                    for previous_point in previous_frame:
                            x_diff = current_point[0] - previous_point[0]
                            y_diff = current_point[1] - previous_point[1]
                            dis= math.sqrt(x_diff**2 + y_diff**2)
                            if dis <=1:
                                stationary_points.append(current_point)
                                break
            print(stationary_points)

            i=0
            stationary_ball_points=[]
            for stationary_point in stationary_points:
                current_stationary_point = stationary_points[i]
                if (math.sqrt(current_stationary_point[0] - stationary_point[0])**2 + 
                (current_stationary_point[1] - stationary_point[1])**2 <= 1):
                    stationary_ball_points.append(stationary_point)
                i += 1


            ball_centers = []
            stationary_ball_withID = []
            for point in stationary_ball_points:
                ball_id = None
                for j in range(len(ball_centers)):
                    x_dis = abs(point[0] - ball_centers[j][0])
                    y_dis = abs(point[1] - ball_centers[j][1])
                    distance = math.sqrt(x_dis**2 + y_dis**2)
                    if distance <= 1:
                        ball_id = j
                if ball_id is None:
                    ball_id = len(ball_centers)
                    ball_centers.append([point[0], point[1]])
                stationary_ball_withID.append([ball_id, point[0], point[1]])
        
            ball_ids = []
            avg_stationary_coordinates = []
            for point in stationary_ball_withID:
                if point[0] not in ball_ids:
                    ball_ids.append(point[0])
            for k in ball_ids:
                x_sum = 0
                y_sum = 0
                count = 0
                for point in stationary_ball_withID:
                    if point[0] == k:
                        x_sum += point[1]
                        y_sum += point[2]
                        count += 1
                avg_x = x_sum / count
                avg_y = y_sum / count
                avg_stationary_coordinates.append([k, avg_x, avg_y])
            print("avg_stationary_coordinate", avg_stationary_coordinates)
        
        filtered_detection_count = 0
        for row in csv_reader[1:]:  # Skip the column top
            if row[0] == "Detection count is":  # Skip the summary
                continue
            if row[2] == "":
                continue
            coordinates = ast.literal_eval(row[2])
            confidences = ast.literal_eval(row[3])
            remaining_coordinates = []
            remaining_confidences = []

            for j in range(len(coordinates)):
                xy = coordinates[j]
                xcoor = 0.5 * xy[0] + 0.5 * xy[2]
                ycoor = 0.5 * xy[1] + 0.5 * xy[3]
                is_stationary = False

                for i in range(len(avg_stationary_coordinates)):
                    x_diff = xcoor - avg_stationary_coordinates[i][1]
                    y_diff = ycoor - avg_stationary_coordinates[i][2]
                    distance = math.sqrt(x_diff**2 + y_diff**2)
                    if distance <= 1:
                        is_stationary = True
                        break

                if not is_stationary:
                    remaining_coordinates.append(xy)
                    remaining_confidences.append(confidences[j])

            if len(remaining_coordinates) == 0:
                row[1] = "0"
                row[2] = ""
                row[3] = ""
            else:
                row[1] = "1"
                row[2] = str(remaining_coordinates)
                row[3] = str(remaining_confidences)
                filtered_detection_count += 1

        csv_reader[-1] = [
            "Detection count is", filtered_detection_count,
            "the ratio is", filtered_detection_count / frame_count,
            "total frames", frame_count,
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_reader)
        print("Detection count after filtering:", filtered_detection_count)

            
                
    print(video_count_multibox, "videos have multiple balls detected")
'''
if __name__ == "__main__":
    main()
