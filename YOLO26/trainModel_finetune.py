from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    model = YOLO(str(ROOT / "yolo26x.pt"))
    model.train(
        data=str(ROOT / "ball.yaml"),
        epochs=30,
        patience=10,
        imgsz=640,
        batch=2,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        warmup_bias_lr=0.0,
        seed=0,
        project=str(ROOT / "runs/train"),
        name="ball_full_x_640_adamw_30ep_warmup0_20261026",
    )


if __name__ == "__main__":
    main()