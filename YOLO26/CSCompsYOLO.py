from ultralytics import YOLO
from pathlib import Path

def main():
    #model = YOLO("yolo26x-seg.pt")
    PROJECT_DIR = Path(__file__).resolve().parent

    best_path = (
        PROJECT_DIR / "runs" / "tennis"
        / "yolo26_run1-10" / "weights" / "best.pt"
    )

    model2 = YOLO(str(PROJECT_DIR / "yolo26x.pt"))

    model = YOLO(str(best_path))
    #match_video = Path.home() / "data/RacketVision/tennis/videos/match100_000.mp4"
    #model(str(match_video), save=True, conf=0.4)
    
    match_video2 = Path.home() / "data/RacketVision/tennis/videos/match208_000.mp4"
    model2(str(match_video2), save=True, conf=0.4, classes = [32])
if __name__ == "__main__":
    main()