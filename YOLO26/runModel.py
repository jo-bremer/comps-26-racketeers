from pathlib import Path
from ultralytics import YOLO
import csv
import json
import math
import dict


PROJECT_DIR = Path(__file__).resolve().parent

def main():
    video_count_multibox = 0
    for i in range(1 , 431):
        model = YOLO(str(PROJECT_DIR / "yolo26x.pt"))
        video = Path.home() / f"data/RacketVision/tennis/videos/match{i}_000.mp4"
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

        with open( output_csv, "w", newline="", encoding="utf-8") as f:
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
         
         
            csv_reader = csv.reader(f)
            center_points = []
            i = 0
            frame_number = 1
              
            for row in islice(csv_reader, 10): # Loop through first 10 .csv rows
                coords = []
                frame_center_points = []
                j = 0
                for xy in row['xyxy']:
                    coords[j] = xy
                    j += 1
                
                for coord in coords:
                    frame_center_points[i] = [0.5*(coord[0] + coord[2]), 0.5*(coord[1] + coord[3])]
                    i += 1
                    
                center_points.append(frame_center_points)
                frame_number += 1
            
            i = 0
            stationary_points = []
            for center_point in center_points:
                current_center_point = center_points[i]
                if (math.sqrt(current_center_point[0] - center_point[0])**2 + 
                   (current_center_point[1] - center_point[1])**2 <= 1):
                    stationary_points.append(center_point)
                    
                i += 1
                
            i = 0
            #stationary_points = []
            stationary_ball_points = []
            for stationary_point in stationary_points:
                current_stationary_point = stationary_points[i]
                if (math.sqrt(current_stationary_point[0] - stationary_point[0])**2 + 
                   (current_stationary_point[1] - stationary_point[1])**2 <= 1):
                    stationary_ball_points.append(stationary_point)
                    
                i += 1
                
            stationary_balls_points = list(dict.fromkeys(stationary_ball_points))
            
            reference_center_points = []
            for ball_entry in stationary_balls_points:
                x_sum = 0
                y_sum = 0
                for center_point in ball_entry:
                    x_sum += center_point[0]
                    y_sum += center_point[1]
                
                reference_center_points.append([x_sum/len(ball_entry), y_sum/len(ball_entry)])  
            
             
                     
                
                    

    print(video_count_multibox, "videos have multiple balls detected")

if __name__ == "__main__":
    main()
