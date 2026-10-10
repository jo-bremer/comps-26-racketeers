from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent



def main():
    model = YOLO(str(PROJECT_DIR / "yolo26x.pt"))
    video = Path.home() / "data/RacketVision/tennis/videos/match5_000.mp4"
    results = model.predict(
        source=str(video),
        save=True,
        conf=0.2,
        classes=[32],
        device=0,
        stream=True,
        project=str(PROJECT_DIR / "runs/detect"),
    )
    frame_count = sum(1 for _ in results)
    print(f"Completed inference on {frame_count} frames.")


if __name__ == "__main__":
    main()
