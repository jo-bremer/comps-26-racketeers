# YOLO26 ball inference

runModel.py is imported with only whitespace cleanup from ~/YOLO26Focus/runModel.
It uses yolo26x.pt, class 32 (sports ball), confidence 0.2 and GPU device 0.
CSCompsYOLO.py remains the existing segmentation starter.

## Setup

The source machine uses Python 3.12 and ultralytics 8.4.174.
Use an environment with compatible CUDA-enabled PyTorch, then install:

```bash
python3 -m pip install ultralytics==8.4.174
```

Place yolo26x.pt beside runModel.py. On the original machine, from the repo root:

```bash
cp ~/YOLO26Focus/yolo26x.pt YOLO26/yolo26x.pt
```

The input video must exist at ~/data/RacketVision/tennis/videos/match5_000.mp4.
Edit video in the script for another input, or device for another device.

## Run

From the repository root:

```bash
python3 YOLO26/runModel.py
```

Annotated output is saved under YOLO26/runs/detect/.
Weights, logs and generated output are ignored by Git. The dataset stays
outside the repository. The original ~/YOLO26Focus directory is preserved.
