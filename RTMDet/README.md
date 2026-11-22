# RTMDet Tennis Ball Detection

## Overview

This folder has all the information for implementing RTMDet for our tennis ball tracking Comps project. I implemented RTMDet-Tiny to detect tennis balls in a dataset from RacketVision and tested different techniques to improve its performance on certain evaluation metrics.

## Model and Dataset

I used RTMDet with MMDetection and fine-tuned the model to detect only one class, which was the tennis ball.

The dataset came from RacketVision, and I used COCO annotations to create bounding boxes around the tennis balls since the original dataset only contained the center coordinates of the ball. I created 15x15 pixel bounding boxes around these coordinates.

## Training

I first trained the model on 640x640 images for 10 epochs with a batch size of 2 and a learning rate of 0.0001.

After looking at the original results, I wanted to see whether increasing the image resolution would improve performance since the tennis ball is a very small object.

I continued training from the original model checkpoint but increased the resolution to 960x960. I used a batch size of 1 because of GPU memory limitations, kept the same learning rate, and trained for another 10 epochs.

The goal was to see whether higher resolution would help the model detect tennis balls that were previously being missed.

## Results

| Metric | Original RTMDet (640) | Improved RTMDet (960) |
|---|---|---|
| Precision | 65.04% | 85.78% |
| Recall | 34.70% | 65.35% |
| mAP50 | 37.4% | 71.8% |
| mAP50-95 | 18.9% | 41.8% |

The improved model performed better on all four metrics. The biggest improvement was in recall, which increased from 34.70% to 65.35%. This was important because the original model was missing many of the tennis balls.

However, I increased the resolution and continued training at the same time, so I cannot say the resolution change alone caused the improvements.

## Mean Distance Error

I also calculated mean distance error (MDE) to measure how far the predicted ball locations were from the actual ball locations.

My original calculation only included detections within 5 pixels of the ground-truth ball. Using this method, the improved model had an MDE of 1.36 pixels.

I recalculated MDE using the highest-confidence prediction above 0.20 for each frame containing an annotated ball, without removing predictions more than 5 pixels away.

This gave me an updated MDE of 15.69 pixels across 1,465 test frames. The other 685 frames did not have a prediction that met the confidence threshold.

This calculation is closer to the method used for YOLO, although the evaluation methods are still not completely identical.

## Motion Filtering

I also experimented with motion filtering to help remove incorrect detections.

The script compares consecutive frames and checks whether there is movement in the area where RTMDet detected a tennis ball.

If there is very little movement, the detection is removed. I used a minimum confidence of 0.05 and required at least 5% of the bounding box area to show movement.

## Motion Tracking

After motion filtering, I also tested a method that used previous ball positions to help track the tennis ball.

The script uses previous detections to estimate where the ball should appear next. If a detection is too far from the expected position, it can be rejected.

The script also allows for missing detections before resetting the tracking information.

These were earlier experiments using a different model checkpoint from the final improved model.

## Folder Organization

- configs/ - Training configurations for the original and improved models.
- evaluation/ - Scripts for calculating precision, recall, F1, and MDE.
- experiments/ - Resolution comparisons and confidence threshold experiments.
- filtering/ - Motion filtering and position-based tracking.
- tracking/ - Video detection and saving tracking predictions.

## Running the Code

I used Python, PyTorch, MMDetection, OpenCV, and NumPy to train and evaluate the models.

Training was done using the MMDetection training tools. The configurations use the RTMDet-Tiny base configuration provided by MMDetection.

The scripts currently contain paths from my Carleton server, so these paths need to be updated when running the code on another computer.

The dataset, trained checkpoints, and large prediction files are not included in this folder.

## Notes

Some scripts were used for earlier experiments and do not all use the same confidence thresholds or matching distances.

Some experiments used a 7-pixel matching distance, while the final evaluation used 5 pixels.

The original matched-only MDE and revised MDE also use different calculation methods, so they should not be directly compared.

## MMDetection Configuration

My training configurations use the RTMDet-Tiny base configuration from MMDetection. This base configuration also depends on other files provided by MMDetection.

To run the training code, MMDetection needs to be installed, and the custom training configurations need to be placed in its RTMDet configuration folder or updated to point to the correct base files.

I included a copy of the RTMDet-Tiny base configuration for reference, but the rest of the MMDetection framework is not included in this repository.
