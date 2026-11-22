# Evaluate the original RTMDet model on the test dataset.
# This script calculates precision, recall, and MDE.
# It also saves the results for each image in a CSV file.
#
# This version uses a confidence threshold of 0.05 and
# counts predictions within 7 pixels as correct matches.

import json
import csv
import cv2
import numpy as np
from pathlib import Path
from mmdet.apis import init_detector, inference_detector

# Set the paths for the model, test images, and results
ROOT = Path("/home/kaurh/RTMDet")
CONFIG = ROOT / "mmdetection/configs/rtmdet/rtmdet_tennis_full.py"
CHECKPOINT = ROOT / "work_dirs/rtmdet_tennis_full/epoch_9.pth"
ANNOTATIONS = ROOT / "annotations_full/ball_test_coco.json"
IMAGE_DIR = ROOT / "frames_full"
OUTPUT = ROOT / "rtmdet_full_evaluation_results.csv"

DISTANCE_THRESHOLD = 7.0
CONFIDENCE_THRESHOLD = 0.05

with open(ANNOTATIONS) as f:
    data = json.load(f)

# Find the actual ball location for each image
ground_truth = {
    ann["image_id"]: (
        ann["bbox"][0] + ann["bbox"][2] / 2,
        ann["bbox"][1] + ann["bbox"][3] / 2
    )
    for ann in data["annotations"]
}

# Load the trained RTMDet model
model = init_detector(str(CONFIG), str(CHECKPOINT), device="cuda:0")

tp = 0
fp = 0
fn = 0
matched_distances = []
rows = []

# Run the model on each test image
for index, image_info in enumerate(data["images"], 1):
    image_path = IMAGE_DIR / image_info["file_name"]

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    result = inference_detector(model, image)
    predictions = result.pred_instances

    boxes = predictions.bboxes.cpu().numpy()
    scores = predictions.scores.cpu().numpy()
    labels = predictions.labels.cpu().numpy()

    candidates = []

    # Only keep ball predictions above the confidence threshold
    for box, score, label in zip(boxes, scores, labels):
        if label != 0 or score < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box
        center_x = float((x1 + x2) / 2)
        center_y = float((y1 + y2) / 2)

        candidates.append((center_x, center_y, float(score)))

    image_id = image_info["id"]
    gt_x, gt_y = ground_truth[image_id]

    # Match at most one prediction to the annotated tennis ball.
    # Select the highest-confidence prediction within 7 pixels.
    matches = []
    for pred_x, pred_y, score in candidates:
        distance = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
        if distance <= DISTANCE_THRESHOLD:
            matches.append((score, distance))

    # Count the best matching prediction as correct
    if matches:
        best_score, best_distance = max(matches, key=lambda x: x[0])
        tp += 1
        fp += len(candidates) - 1
        matched_distances.append(best_distance)
    else:
        best_score = None
        best_distance = None
        fp += len(candidates)
        fn += 1

    # Save the results for this image
    rows.append({
        "image": image_info["file_name"],
        "gt_x": gt_x,
        "gt_y": gt_y,
        "predictions": len(candidates),
        "matched": int(best_distance is not None),
        "matched_distance": best_distance,
        "matched_confidence": best_score
    })

    if index % 250 == 0:
        print(f"Processed {index}/{len(data['images'])} images", flush=True)

# Calculate the final precision, recall, and MDE
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
mde = float(np.mean(matched_distances)) if matched_distances else float("nan")

# Write the results to a CSV file
with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nRTMDet Full Test Evaluation")
print("True positives:", tp)
print("False positives:", fp)
print("False negatives:", fn)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"MDE (matched detections): {mde:.2f} pixels")
print("Results saved:", OUTPUT)
