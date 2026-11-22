# Run RTMDet on a tennis video and save the detections.
# This script uses the original trained model to find tennis balls
# in each frame and draws a box around the predictions.
# Only predictions with confidence of at least 0.20 are shown.

import cv2
from pathlib import Path
from mmdet.apis import init_detector, inference_detector

# Set the paths for the model and video files
ROOT = Path.home() / "RTMDet"

CONFIG = ROOT / "mmdetection/configs/rtmdet/rtmdet_tennis_full.py"
CHECKPOINT = ROOT / "work_dirs/rtmdet_tennis_full/epoch_9.pth"

INPUT_VIDEO = ROOT / "match10_000.mp4"
OUTPUT_VIDEO = ROOT / "RTMDet_output_match10_000.mp4"

CONFIDENCE_THRESHOLD = 0.20

# Load the trained RTMDet model
print("Loading RTMDet model...", flush=True)
model = init_detector(str(CONFIG), str(CHECKPOINT), device="cuda:0")

# Open the video and get its frame information
cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open input video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video")

print(f"Processing {total} frames at {width}x{height}, {fps:.2f} FPS", flush=True)

frame_number = 0

# Go through each frame and detect tennis balls
while True:
    success, frame = cap.read()

    if not success:
        break

    result = inference_detector(model, frame)
    predictions = result.pred_instances

    boxes = predictions.bboxes.cpu().numpy()
    scores = predictions.scores.cpu().numpy()
    labels = predictions.labels.cpu().numpy()

    # Only draw boxes for ball detections above the threshold
    for box, score, label in zip(boxes, scores, labels):
        if label != 0 or score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Ball {score:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "RTMDet",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        3
    )

    # Save the frame with the detected balls
    writer.write(frame)
    frame_number += 1

    if frame_number % 50 == 0:
        print(f"Processed {frame_number}/{total} frames", flush=True)

cap.release()
writer.release()

print("Finished!")
print("Output saved to:", OUTPUT_VIDEO)
