from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    model = YOLO(str(ROOT / "yolo26x.pt"))

    model.train(
        data=str(ROOT / "ball50.yaml"),
        epochs=1,
        imgsz=640,
        batch=2,
        device=0,
        project=str(ROOT / "runs/train"),
        name="ball50_demo",
    )


if __name__ == "__main__":
    main()