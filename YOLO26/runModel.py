from pathlib import Path
from ultralytics import YOLO
import csv
import json


PROJECT_DIR = Path(__file__).resolve().parent

def main():
    model = YOLO(str(PROJECT_DIR / "yolo26x.pt"))
    video = Path.home() / "data/RacketVision/tennis/videos/match5_000.mp4"
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
        detection_count =0

        frame_number = 0 #assuming we start from frame 1
        for result in results:
            frame_count += 1
            boxes = result.boxes

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
    print("the Detection count is ", detection_count, "the ratio is", detection_count/frame_count)
    print(f"Completed inference on {frame_count} frames.")

if __name__ == "__main__":
    main()
