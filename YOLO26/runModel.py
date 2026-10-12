from pathlib import Path
from ultralytics import YOLO

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
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "frame_number",
            "visibility",
            "xyxy",
            "confidence",
        ])
    print(f"Completed inference on {frame_count} frames.")

    frame_count = 0
    detection_count = 0
    box_count = 0

    frame_number = 0 #assuming we start from frame 1
    for result in results:
        frame_number += 1
        boxes = result.boxes

        if len(boxes) == 0:
            writer.writerow([
                        frame_number,
                        0,
                        "",
                        "",
                    ])
            continue

        boxesPerFrame = []
        confPerFrame = []
        for box in boxes:
            box_count +=1 

            confidence = box.conf.item()


            confidence 


        writer.writerow([
                        frame_number,
                        1,
                        "",
                        ,])


        
            


if __name__ == "__main__":
    main()
