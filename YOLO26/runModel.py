from pathlib import Path
from ultralytics import YOLO
import csv
import json
import math
import itertools

#import dict


PROJECT_DIR = Path(__file__).resolve().parent

def main():
    video_count_multibox = 0
    #for i in range(1 , 431):
    model = YOLO(str(PROJECT_DIR / "yolo26x.pt"))
    video = Path.home() / f"data/RacketVision/tennis/videos/match{5}_000.mp4"
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
                print(coords)
                print(new_coords)
                for coord in new_coords:
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
            #stationary_points = [] all stationary points in a list from the first 10 frames
            stationary_ball_points = [] #stationary_ball_points each entry is stationary points for one ball (only format difference)
            for stationary_point in stationary_points:
                current_stationary_point = stationary_points[i]
                if (math.sqrt(current_stationary_point[0] - stationary_point[0])**2 + 
                (current_stationary_point[1] - stationary_point[1])**2 <= 1):
                    stationary_ball_points.append(stationary_point)
                    
                i += 1
                
            stationary_balls_points = list(dict.fromkeys(stationary_ball_points))
            
            reference_center_points = [] # each entry is the average of one ball
            for ball_entry in stationary_balls_points:
                x_sum = 0
                y_sum = 0
                for center_point in ball_entry:
                    x_sum += center_point[0]
                    y_sum += center_point[1]
                
                reference_center_points.append([x_sum/len(ball_entry), y_sum/len(ball_entry)])  
                
                for row in csv_reader: # Loop through first 10 .csv rows
                    j=0
                    for xy in row['xyxy']: #[[x, y, x, y], [x,y,x,y]]
                        xcoor = 1/2 *xy[0]+1/2 *xy[2] 
                        ycoor = 1/2 *xy[1]+1/2 *xy[3] 
                        curr_center = [xcoor,ycoor]
                        for i in len(reference_center_points):
                            distance = math.sqrt( (reference_center_points[i][0] - curr_center[0])**2 +(reference_center_points[i][1] - curr_center[1])**2)
                            if distance <= 1:
                                xy[j] = ""
                                row['confidence'] = ""
                                if len(xy) == 0:
                                    row['visibility'] = 0
                                j -= i
                        j += 1
                with open(output_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(csv_reader)       

                    
            
                
    print(video_count_multibox, "videos have multiple balls detected")

if __name__ == "__main__":
    main()
