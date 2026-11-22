# RTMDet 960 Evaluation
#
# This script evaluates the improved RTMDet model on the test set.
# It compares the predicted ball locations with the ground-truth labels.
# It calculates precision, recall, F1 score, and mean distance error.
#
# A prediction is counted as correct if it is within 5 pixels
# of the actual ball location.
# MDE here only includes predictions that were matched successfully.
#
# The separate recalculate_mde.py script calculates MDE without
# the 5-pixel distance cutoff.

import json
import pickle
import numpy as np
from pathlib import Path

# Find the prediction and annotation files
root = Path.home() / "RTMDet"

prediction_file = root / "work_dirs/rtmdet_tennis_960/test_predictions.pkl"
annotation_file = root / "annotations_full/ball_test_coco.json"

# Minimum confidence and distance allowed for a correct match
CONFIDENCE = 0.20
MAX_DISTANCE = 5.0

# Load the predictions and ground-truth annotations
with open(prediction_file, "rb") as f:
    predictions = pickle.load(f)

with open(annotation_file, "r") as f:
    annotations = json.load(f)

# Organize ground-truth boxes by image ID
ground_truth = {}

for image in annotations["images"]:
    ground_truth[image["id"]] = []

for ann in annotations["annotations"]:
    if ann.get("iscrowd", 0):
        continue

    x, y, w, h = ann["bbox"]
    center = (x + w / 2, y + h / 2)

    ground_truth[ann["image_id"]].append(center)

# Keep track of correct detections, incorrect detections, and missed balls
TP = 0
FP = 0
FN = 0
distances = []

# Go through each test image and compare predictions with labels
for result in predictions:
    image_id = result["img_id"]
    gt_centers = ground_truth.get(image_id, [])

    instances = result["pred_instances"]

    boxes = np.asarray(instances["bboxes"])
    scores = np.asarray(instances["scores"])
    labels = np.asarray(instances["labels"])

    # Keep ball predictions at our selected threshold
    keep = (scores >= CONFIDENCE) & (labels == 0)

    boxes = boxes[keep]
    scores = scores[keep]

    # Process predictions from highest confidence to lowest
    order = np.argsort(-scores)
    boxes = boxes[order]

    matched_gt = set()

    for box in boxes:
        x1, y1, x2, y2 = box

        # Find the center of the predicted bounding box
        pred_center = (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

        best_distance = float("inf")
        best_index = None

        # Find the closest ground-truth ball that has not been matched
        for i, gt_center in enumerate(gt_centers):
            if i in matched_gt:
                continue

            distance = np.hypot(
                pred_center[0] - gt_center[0],
                pred_center[1] - gt_center[1]
            )

            if distance < best_distance:
                best_distance = distance
                best_index = i

        # Count the prediction as correct if it is within 5 pixels
        if best_index is not None and best_distance <= MAX_DISTANCE:
            TP += 1
            matched_gt.add(best_index)
            distances.append(best_distance)
        else:
            FP += 1

    # Count any ground-truth balls that were not detected
    FN += len(gt_centers) - len(matched_gt)

# Calculate the final evaluation metrics
precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)
# MDE only includes distances from successfully matched predictions
mde = np.mean(distances) if distances else float("nan")

print("\n========== RTMDet 960 TEST RESULTS ==========")
print("Images:", len(predictions))
print("Confidence threshold:", CONFIDENCE)
print("Matching radius:", MAX_DISTANCE, "pixels")
print("---------------------------------------------")
print("True positives:", TP)
print("False positives:", FP)
print("False negatives:", FN)
print("---------------------------------------------")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"MDE:       {mde:.2f} pixels (matched only)")
print("=============================================\n")
