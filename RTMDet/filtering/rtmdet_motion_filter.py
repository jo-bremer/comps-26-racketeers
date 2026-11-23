# RTMDet Motion Filtering
#
# This script detects tennis balls in a video using RTMDet.
# It compares each frame with the previous frame to check for movement.
# Detections with very little movement are filtered out.
# The remaining detections are drawn on the video and saved.
#
# This was an earlier experiment using the epoch 2 model.
# Update the file paths below before running it on another computer.

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

config = "/home/kaurh/RTMDet/mmdetection/configs/rtmdet/rtmdet_tennis.py"
checkpoint = "/home/kaurh/RTMDet/work_dirs/rtmdet_tennis/epoch_2.pth"
video_path = "/home/kaurh/RTMDet/match100_000.mp4"
output_path = "/home/kaurh/RTMDet/tennis_motion_filtered.mp4"

# Load the trained RTMDet model
model = init_detector(config, checkpoint, device="cuda:0")

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Keep track of the previous frame and detection counts
previous_gray = None
total_candidates = 0
kept_candidates = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = inference_detector(model, frame)
    predictions = result.pred_instances

    scores = predictions.scores.cpu().numpy()
    boxes = predictions.bboxes.cpu().numpy()

    # Compare the current frame with the previous one
    if previous_gray is not None:
        difference = cv2.absdiff(previous_gray, gray)
        difference = cv2.GaussianBlur(difference, (5, 5), 0)

        _, motion_mask = cv2.threshold(
            difference, 20, 255, cv2.THRESH_BINARY
        )

        # Check each detected box for movement
        for score, box in zip(scores, boxes):
            if score < 0.05:
                continue

            total_candidates += 1

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            motion_region = motion_mask[y1:y2, x1:x2]

            if motion_region.size == 0:
                continue

            # Find how much of the box contains movement
            motion_ratio = np.count_nonzero(motion_region) / motion_region.size

            if motion_ratio < 0.05:
                continue

            kept_candidates += 1

            # Draw a green box around detections we keep
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Ball {score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    writer.write(frame)
    previous_gray = gray

cap.release()
writer.release()

# Show how many detections were kept or removed
print("Motion-filtered video saved:", output_path)
print("Candidate detections:", total_candidates)
print("Detections kept:", kept_candidates)
print("Detections removed:", total_candidates - kept_candidates)
