# YOLO26 tennis-ball workflow

The main workflow runs from this folder without `comps-training-jobs`, launch wrappers, or another person's home directory. Edit `project_config.json` when moving to a different computer. Data and checkpoints are required inputs; they are not bundled with the source code.

## Files that make up the main workflow

| File | Purpose |
| --- | --- |
| `project_config.json` | Data, pretrained weights, saved checkpoint aliases, settings, and output paths |
| `project_config.py` | Resolve paths and generate dataset YAML; inspect inputs without loading a model |
| `runner.py` | Prepare YOLO images/labels from videos and annotations, or verify existing prepared data |
| `trainModel_ft640.py` | Train at image size 640 |
| `trainModel_ft1280.py` | Train at image size 1280 |
| `runModel.py` | Video inference, raw/filtered CSVs, optional annotated videos |
| `evaluate.py` | Precision, recall, mAP50, mAP50–95, MDE, and optional six-cell comparison |
| `evaluate_pretrained_stationary.py` | Raw-versus-filtered check on the historical pilot validation set |
| `requirements.txt`, `environment_reference.json` | Application version pins and the observed original environment |
| `README.md` | This reproduction guide |

`ball.yaml` and `ball50.yaml` are generated from the configuration; do not maintain a second set of paths in them. Team scripts such as `runModelMatthew.py` and `CSCompsYOLO.py`, their configuration files, and previous experiment outputs are retained separately from the main workflow. They are not dependencies of these commands.

## Setup and required inputs

Use Python 3.12. The original server environment is recorded in `environment_reference.json`. Install a PyTorch/torchvision build compatible with the destination GPU and driver, then install `requirements.txt`. The recorded CUDA build is `torch 2.11.0+cu128` / `torchvision 0.26.0+cu128`; CPU or different GPU builds can change runtime and results. No packages need reinstalling on the already configured server.

```sh
python3 -m venv .venv
. .venv/bin/activate
# Install the appropriate torch/torchvision build for your machine first.
python -m pip install -r requirements.txt
```

Change these fields in `project_config.json`:

- `data.raw_root`: original tennis dataset containing `videos/*.mp4`, `all/match*/csv/*_ball.csv`, and official `info/train.json`, `info/val.json`, `info/test.json`.
- `data.prepared_root` and `data.pilot_root`: separate locations for the full and historical first-50 prepared datasets.
- `model.pretrained`: the original `yolo26x.pt` checkpoint. This file is required for training and default inference.
- `checkpoints`: your saved fine-tuned checkpoints, required for evaluating existing models but not for training new ones.
- `runtime.device`: `0` for the original GPU or `"cpu"` for CPU execution. Training on CPU will be much slower.

All relative paths, including CLI `--source` and `--weights`, start at the **configuration file's folder**, regardless of the shell's working directory. Defaults match the existing `/home/gonga2/comps-26-racketeers/YOLO26` layout.

```sh
cd /home/gonga2/comps-26-racketeers/YOLO26
python3 project_config.py --check
```

This checks required paths and reports installed package versions; it does not load a model. Missing saved checkpoint aliases are reported separately and do not prevent training new models. Package installation remains your environment setup step.

## 1. Prepare or reuse the dataset

For existing prepared datasets on this server:

```sh
python3 runner.py --dataset full --reuse
python3 runner.py --dataset pilot --reuse
```

For a new destination that does not yet exist, omit `--reuse`:

```sh
python3 runner.py --dataset full
python3 runner.py --dataset pilot
```

Preparation does **not** start training. It uses official video split membership, includes annotated frames only, saves original-size JPEGs at quality 95, and produces one class-0 label per visible ball. Labels are 15×15-pixel boxes centered on the annotated point and clipped at image borders. Visibility 0 produces an empty label file. An unannotated frame is excluded, not relabeled as negative. Frame indices in the manifest start at zero.

The frame extraction and separate label-writing stages in `runner.py` are adapted from Matthew’s preparation code in `runModelMatthew.py`. The integration preserves the existing 15×15 clipped-label recipe, filenames, split membership and metadata. It reads each annotation CSV once, then uses sequential `capture.read()` calls. This retrieves every frame up to the last annotation and may take more work than the previous selective retrieval; timing has not been measured. Matthew’s original file remains unchanged, and training stays in the separate training scripts.

Historical membership: full = 350/38/43 train/val/test videos and 17,494/1,900/2,150 annotated images. Pilot = match1 through match50, 41/4/5 videos and 2,049/200/250 images. `--reuse` verifies split membership, paired files, label structure, and recorded counts; it does not re-decode images or prove byte-for-byte identity with raw videos.

## 2. Train at 640 and 1280

Use **new run names** so previous results remain intact:

```sh
python3 trainModel_ft640.py --name full640_reproduction
python3 trainModel_ft1280.py --name full1280_reproduction
```

Each creates `runs/train/<name>/`, including library training outputs, arguments, `results.csv`, and `weights/best.pt` / `weights/last.pt`. Existing output names are refused. The configured historical names already exist on the original server, so running without `--name` there intentionally stops.

The full-data defaults reproduce the recorded recipe of the completed runs: pretrained `yolo26x.pt`, 30-epoch limit, patience 10, batch 2, AdamW, initial learning rate 0.001, warmup 3 epochs, warmup bias learning rate 0, seed 0. They restore the completed-run settings rather than copying the later 640-script changes (`batch=-1`, explicit `nms=False`). Remaining library defaults depend on the installed version.

For the pilot experiment:

```sh
python3 trainModel_ft640.py --dataset pilot --name pilot640_reproduction
python3 trainModel_ft1280.py --dataset pilot --epochs 1 --name pilot1280_one_epoch_reproduction
```

Pilot warmup bias learning rate is 0.1. The old pilot-1280 run saved only one epoch despite its `30ep` directory name; the second command reproduces that epoch limit. Omit `--epochs 1` to run the intended 30-epoch pilot experiment, which is a new experiment rather than a reproduction of that saved run.

## 3. Evaluate saved models

Evaluate existing historical aliases:

```sh
python3 evaluate.py --model full640 --imgsz 640
python3 evaluate.py --model full1280 --imgsz 1280
```

Evaluate a **newly trained** checkpoint using its actual path (aliases continue to point at historical checkpoints until edited):

```sh
python3 evaluate.py --weights runs/train/full640_reproduction/weights/best.pt --imgsz 640
python3 evaluate.py --weights runs/train/full1280_reproduction/weights/best.pt --imgsz 1280
```

Validation is the default. Use `--split test` for a separately reported held-out test result; tuning choices should be settled before interpreting the final test. Use `--dataset pilot` only when you specifically want the small pilot evaluation set. To reproduce the historical comparison structure—pilot640, full640, full1280, each evaluated at both resolutions on full validation:

```sh
python3 evaluate.py --matrix
```

Each run has a new folder under `runs/evaluation/`, with library plots/prediction labels, `result.json`, a one-row `metrics.csv`, and per-image `center_errors.csv`. Matrix mode also writes a `summary.csv`, updated after each successful cell. You can combine the one-row CSV files manually for your own table.

Metric definitions:

- Precision, recall, mAP50 and **mAP50–95** come from `model.val`. The standard last metric is not mAP50–90.
- MDE selects the highest-confidence prediction at confidence ≥0.25 in each positive frame and averages the Euclidean center distance in **original-image pixels**. It uses the prepared box center. There is no IoU or distance matching cutoff for MDE.
- Positive frames with no qualifying prediction are excluded from the MDE mean and counted as misses. Report scored frames, misses, and prediction coverage alongside MDE. Coverage is not IoU-based recall. Predictions on negative frames are counted separately.
- The fixed confidence used for MDE is different from the library's reported P/R operating point. A lower MDE alone does not prove a better detector.

`nms: null` means **omit the NMS argument**, as in the historical evaluator. It does not mean NMS is disabled. `result.json` records the actual validator's NMS and end-to-end flags. `--nms on`, `--nms off`, and `--nms default` are explicit experiment overrides; changing this setting may change results. The script checks that the validator processed every manifest image.

## 4. Inference, CSV filtering, and video output

```sh
python3 runModel.py --source ../../data/RacketVision/tennis/videos/match1_000.mp4
python3 runModel.py --source ../../data/RacketVision/tennis/videos/match1_000.mp4 --weights runs/train/full640_reproduction/weights/best.pt --save-video
```

Omitting `--source` processes the whole configured video folder. Default inference uses the pretrained checkpoint at confidence 0.2 and image size 640. Class selection comes from model metadata: COCO sports-ball class 32 or fine-tuned ball class 0; `--class-id` overrides ambiguous metadata.

Outputs go to a new `runs/inference/<timestamp>/` folder: `raw/*.csv`, `filtered/*.csv`, and `summary.json`. CSV frame numbers start at **one**, matching the original exporter, and keep its summary footer. Annotated videos show **raw detections**, before the stationary filter. To reproduce the old `getVideo.py` confidence setting, add `--conf 0.05` and explicitly choose the intended checkpoint.

The heuristic identifies near-stationary centers within the first ten frames, skipping empty frames when comparing detected centers, then removes detections within a one-pixel radius of the learned locations throughout the video. It is removal-only, can remove a real ball, and does not replace the detector's internal postprocessing. The original implementation overwrote raw CSVs; this version preserves both versions for inspection. `summary.json` records detection/multiple-detection counts, which are descriptive counts, not ground-truth precision or recall.

## 5. Check whether the stationary filter helps

Prepare/reuse the pilot dataset first, then run:

```sh
python3 evaluate_pretrained_stationary.py
```

This reproduces the original quick-check protocol: pretrained model, 640 input, confidence 0.2, first-50 intersection with official validation (four videos; 200 annotated frames, 193 positive), IoU matching threshold 0.5. It compares raw and filtered fixed-threshold P/R/F1 and reports target detections lost. Its CSV frame indices start at **zero** to align with annotations. `--weights` permits a fine-tuned checkpoint as a new comparison. Outputs go to `runs/stationary_evaluation/`.

These metrics are separate from `model.val`'s AP computation. This check does not establish an improvement on the full held-out test set, and filtering is not automatically applied by `evaluate.py`.

## Reproduction scope and retained evidence

This workflow recreates the main kinds of outputs and preserves the key protocols. Exact numerical reproduction is not guaranteed across hardware, package changes, random training behavior, or a differently prepared dataset. Keep raw data and official splits, initial and trained checkpoints, run arguments, `results.csv`, manifests, and evaluation results if you want to defend the original results.

The November 15 refactor and its recoverable cleanup are recorded in Obsidian and a manifest under `/home/gonga2/repository-cleanup-backups/`. Earlier source versions, logs, and experiment evidence remain recoverable. No training or evaluation was executed as part of that cleanup. The November 10 Downloads learning package explains the earlier source snapshot; use this README and current code for the revised commands.
