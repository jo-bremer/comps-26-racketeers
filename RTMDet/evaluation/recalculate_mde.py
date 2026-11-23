# RTMDet MDE Evaluation
#
# This script checks how close our RTMDet predictions are
# to the actual tennis ball locations.
#
# The first calculation uses a 5-pixel matching distance.
# It calculates precision, recall, F1, and MDE for matched balls.
#
# The second calculation finds the highest-confidence ball
# prediction in each frame and calculates its distance from
# the actual ball without using the 5-pixel cutoff.
#
# Frames without a prediction above 0.20 are not included
# in the second MDE calculation.

import json
import pickle
import numpy as np
from pathlib import Path

root = Path.home() / "RTMDet"

prediction_file = root / "work_dirs/rtmdet_tennis_960/test_predictions.pkl"
annotation_file = root / "annotations_full/ball_test_coco.json"

CONFIDENCE = 0.20
MAX_DISTANCE = 5.0

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

TP = 0
FP = 0
FN = 0
distances = []

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

        pred_center = (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

        best_distance = float("inf")
        best_index = None

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

        if best_index is not None and best_distance <= MAX_DISTANCE:
            TP += 1
            matched_gt.add(best_index)
            distances.append(best_distance)
        else:
            FP += 1

    FN += len(gt_centers) - len(matched_gt)

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)
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

# Calculate MDE again without the 5-pixel distance limit.
# This helps us compare the results with the other models.
# We only use frames that have a ground-truth ball and a
# prediction above our confidence threshold.
all_distances = []
no_prediction = 0

for result in predictions:
    gt_centers = ground_truth.get(result["img_id"], [])
    if not gt_centers:
        continue

    instances = result["pred_instances"]
    boxes = np.asarray(instances["bboxes"])
    scores = np.asarray(instances["scores"])
    labels = np.asarray(instances["labels"])

    valid = np.where((scores >= CONFIDENCE) & (labels == 0))[0]
    if len(valid) == 0:
        no_prediction += 1
        continue

    best = valid[np.argmax(scores[valid])]
    x1, y1, x2, y2 = boxes[best]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    error = min(
        np.hypot(cx - gx, cy - gy)
        for gx, gy in gt_centers
    )
    all_distances.append(error)

print("\n===== REVISED RTMDet MDE =====")
print("Frames included:", len(all_distances))
print("Frames without qualifying predictions:", no_prediction)
print("MDE without 5-pixel cutoff:",
      round(float(np.mean(all_distances)), 4)
      if all_distances else "N/A")
print("=============================")
