from pathlib import Path
from ultralytics import YOLO
import csv
import json
import math
import itertools
import ast

#import dict


PROJECT_DIR = Path(__file__).resolve().parent

def main():
    video_count_multibox = 0
    model = YOLO(str(PROJECT_DIR / "yolo26x.pt"))
    video_dir = Path.home() / "data/RacketVision/tennis/videos"
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

if __name__ == "__main__":
    main()
