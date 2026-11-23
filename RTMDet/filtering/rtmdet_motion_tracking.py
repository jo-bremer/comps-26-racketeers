# RTMDet Motion Tracking
#
# This script uses RTMDet to find tennis balls in a video.
# It first checks for movement to help remove false detections.
# It also uses previous ball positions to estimate where the ball
# should be in the next frame.
#
# The best detection is drawn on the video and saved.
# This was an earlier experiment using the epoch 2 model.
# Update the paths below before running this script.

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

config = "/home/kaurh/RTMDet/mmdetection/configs/rtmdet/rtmdet_tennis.py"
checkpoint = "/home/kaurh/RTMDet/work_dirs/rtmdet_tennis/epoch_2.pth"
video_path = "/home/kaurh/RTMDet/match100_000.mp4"
output_path = "/home/kaurh/RTMDet/tennis_motion_tracking.mp4"

# Load the RTMDet model
model = init_detector(config, checkpoint, device="cuda:0")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open tennis video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video")

# Store previous frames and ball positions for tracking
previous_gray = None
last_position = None
previous_position = None
missing_frames = 0

total_candidates = 0
motion_candidates = 0
selected_detections = 0

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = inference_detector(model, frame)
    predictions = result.pred_instances

    scores = predictions.scores.cpu().numpy()
    boxes = predictions.bboxes.cpu().numpy()

    # Compare frames to find areas with movement
    if previous_gray is not None:
        difference = cv2.absdiff(previous_gray, gray)
        difference = cv2.GaussianBlur(difference, (5, 5), 0)
        _, motion_mask = cv2.threshold(
            difference, 20, 255, cv2.THRESH_BINARY
        )

        candidates = []

        # Check each possible ball detection
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

            motion_ratio = (
                np.count_nonzero(motion_region)
                / motion_region.size
            )

            if motion_ratio < 0.05:
                continue

            motion_candidates += 1

            center = np.array([
                (x1 + x2) / 2,
                (y1 + y2) / 2
            ])

            # Confidence and motion are both useful signals.
            # Give each detection a score using confidence and movement
            candidate_score = float(score) + 0.05 * motion_ratio

            # Estimate the next ball position using previous detections
            if last_position is not None and missing_frames < 4:
                predicted_position = last_position.copy()

                if previous_position is not None:
                    velocity = last_position - previous_position
                    predicted_position += velocity * (missing_frames + 1)

                distance = np.linalg.norm(
                    center - predicted_position
                )

                # Reject large, sudden jumps.
                max_distance = 110 * (missing_frames + 1)

                # Ignore detections that are too far away
                if distance > max_distance:
                    continue

                candidate_score -= 0.001 * distance

            candidates.append(
                (candidate_score, center, (x1, y1, x2, y2), score)
            )

        # Choose the best remaining ball detection
        if candidates:
            best = max(candidates, key=lambda item: item[0])

            _, center, box, score = best
            x1, y1, x2, y2 = box

            selected_detections += 1

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.circle(
                frame,
                tuple(center.astype(int)),
                5,
                (0, 255, 255),
                -1
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

            # Save the current position for the next frame
            previous_position = last_position
            last_position = center
            missing_frames = 0

        else:
            missing_frames += 1

            # Reset tracking if the ball has been missing too long
            if missing_frames >= 4:
                last_position = None
                previous_position = None

    writer.write(frame)
    previous_gray = gray

    if frame_number % 50 == 0:
        print("Processed frames:", frame_number, flush=True)

cap.release()
writer.release()

# Print the results after processing the video
print("Output:", output_path)
print("Total candidates:", total_candidates)
print("Passed motion filter:", motion_candidates)
print("Frames with selected detections:", selected_detections)
print("Total frames:", frame_number)
