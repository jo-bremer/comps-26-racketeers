# TennisVision: Tennis Ball Tracking Model Comparison

This repository has the code and experiments for our groups comps project. **Tennis Vision: An Analysis and Comparison of Machine Learning Architectures for Tennis Ball-Tracking**.

We compared 3 different models to how well the preformed in tracking balls in tennis videos using the tennis portion of the [RacketVision dataset](https://doi.org/10.48550/arXiv.2511.17045): **YOLO26**, **TrackNetV5**, and **RTMDet-Tiny**. The models use different ways to detect the ball, train the models, and evaluate how well they do. The code is organized by each individual model to see how well they preform. 

## Repository layout

```text
comps-26-racketeers/
├── README.md                    # This overview
├── config.py                    # Shared video and annotation settings
├── requirements.txt             # Shared Python dependency list
├── annotations/                # Read and visualize ball annotations
├── YOLO26/                     # YOLO26 preprocessing, training, and evaluation
├── tracknet/                   # TrackNetV5 SDK integration and supporting scripts
└── RTMDet/                     # RTMDet configurations, evaluation, and experiments
```

The repository also has`.gitignore` files and editor config under `.vscode/`.

## Data

The full dataset is from RacketVision and can be downloaded from the tennis videos and their annotations/split files separately. The models prepare their data in their formats required by each model. Each model has its own workflow and way of running depending on how they are formatted to wrok.

A starting point for shared Python packages is:

```bash
python3 -m pip install -r requirements.txt
```

## File guide

### Shared annotation tools and settings

| File | Purpose |
| --- | --- |
| `config.py` | Shared constants for videos, FPS, output format, and a five-pixel prediction radius. |
| `requirements.txt` | Shared Python package requirements. |
| `annotations/read_annotations.py` | Reads ball-position annotations for use by other tools. |
| `annotations/annotation_visualizer.py` | Draws annotated ball centers on video frames and saves an annotated video. |

### Files of other models
Please refer the README's in each models folder to see what the files do and how they are configured. 

## Results overview
Our paper compared each ball tracking model on recision, recall, mAP@50, and mean distance error (MDE). The figures below described the results seen from each model after finetuning and training each model.

| Model | Precision | Recall | mAP@50 | MDE (pixels) |
| --- | ---: | ---: | ---: | ---: |
| YOLO26x | 93.3% | 88.4% | 93.2% | 6.55 |
| TrackNetV5 | 73.9% | 70.8% | 66.8% | 12.44 |
| RTMDet-Tiny | 85.8% | 65.4% | 71.8% | 15.69 |


## Suggested workflow

1. Download RacketVision tennis videos, labels, and official split information.
2. Choose one model folder at a time.
3. Read the model-specific README or SDK documentation and install required downloads.
4. Update data paths, checkpoint locations, and configuration values for your environment.
5. Run that model's preprocessing, training/inference, and evaluation as stated in their folder.
6. Compare the outputs only after checking that evaluation metrics for the modelS

## Project credits

This repository was made in a project related to Carleton College Image Segmentation Comps and was created by: 

**Harshdeep Kaur, Andrew Gong, Matthew Hall, and Jo Bremer**.

