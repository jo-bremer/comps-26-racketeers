"""Evaluate saved YOLO26 checkpoints; this script never trains a model.

One result: python3 evaluate.py --model full640 --imgsz 640
Own weights: python3 evaluate.py --weights runs/train/example/weights/best.pt --imgsz 640
Six comparisons: python3 evaluate.py --matrix
The default dataset is full and the default split is val.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from math import hypot, isfinite
from pathlib import Path

from project_config import load_config, resolve_path, dataset_root, write_dataset_yaml


MATRIX_MODELS = ("pilot640", "full640", "full1280")
MATRIX_SIZES = (640, 1280)


def read_boxes(path):
    """Read class, normalized center x/y, width, height, [confidence]."""
    if not path.exists():
        return []  # A missing prediction file means no detections.
    return [list(map(float, line.split()))
            for line in path.read_text().splitlines() if line.strip()]


def check_boxes(boxes, path, predictions=False):
    """Reject malformed labels explicitly, including when Python runs with -O."""
    if not predictions and len(boxes) > 1:
        raise ValueError(f"Expected at most one labeled ball: {path}")
    fields = 6 if predictions else 5
    for box in boxes:
        if len(box) != fields or not all(isfinite(value) for value in box):
            raise ValueError(f"Expected {fields} finite YOLO fields: {path}")
        if box[0] != 0 or not all(0 <= value <= 1 for value in box[1:5]):
            raise ValueError(f"Expected class 0 and normalized coordinates: {path}")
        if not predictions and (box[3] <= 0 or box[4] <= 0):
            raise ValueError(f"Ground-truth box dimensions must be positive: {path}")
        # Clipping can produce a zero-area prediction; do not add a new MDE size gate.
        if predictions and not 0 <= box[5] <= 1:
            raise ValueError(f"Prediction confidence must be between 0 and 1: {path}")


def calculate_mde(image_paths, label_folder, prediction_folder, confidence):
    """Score the top-confidence center in original pixels; count misses separately."""
    from PIL import Image

    if not 0 <= confidence <= 1:
        raise ValueError("MDE confidence must be between 0 and 1")
    errors, frame_rows = [], []
    positive_frames = missed_frames = negative_false_positives = 0
    for image_path in image_paths:
        filename = image_path.stem + ".txt"
        label_path = label_folder / filename
        if not label_path.exists():
            raise FileNotFoundError(f"Missing ground-truth label: {label_path}")
        ground_truth = read_boxes(label_path)
        prediction_path = prediction_folder / filename
        predictions = read_boxes(prediction_path)
        check_boxes(ground_truth, label_path)
        check_boxes(predictions, prediction_path, predictions=True)
        predictions = [box for box in predictions if box[5] >= confidence]
        prediction = max(predictions, key=lambda box: box[5], default=None)
        with Image.open(image_path) as image:
            width, height = image.size
        gt_x = gt_y = pred_x = pred_y = distance = None
        if prediction is not None:
            pred_x, pred_y = prediction[1] * width, prediction[2] * height
        if ground_truth:
            positive_frames += 1
            gt_x, gt_y = ground_truth[0][1] * width, ground_truth[0][2] * height
            if prediction is None:
                missed_frames += 1
            else:
                distance = hypot(pred_x - gt_x, pred_y - gt_y)
                errors.append(distance)
        elif prediction is not None:
            negative_false_positives += 1
        frame_rows.append({"image": image_path.name, "gt_x": gt_x, "gt_y": gt_y,
                           "pred_x": pred_x, "pred_y": pred_y, "distance_px": distance,
                           "prediction_confidence": prediction[5] if prediction else None})
    return {
        "mde_confidence": confidence,
        "mde_px": sum(errors) / len(errors) if errors else None,
        "mde_scored_frames": len(errors),
        "positive_frames": positive_frames,
        "missed_positive_frames": missed_frames,
        "prediction_coverage": len(errors) / positive_frames if positive_frames else None,
        "negative_frames_with_prediction": negative_false_positives,
    }, frame_rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(config, base, dataset_name, split):
    """Check the prepared dataset before loading a model; preserve folder-based validation."""
    root = dataset_root(config, base, dataset_name)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = [row for row in manifest if row["split"] == split]
    if not selected:
        raise ValueError(f"No {split} images in {manifest_path}")
    names = [row["image"] for row in selected]
    if any(not isinstance(name, str) or Path(name).name != name or name in ("", ".", "..")
           for name in names):
        raise ValueError("Manifest image names must be plain filenames")
    if len(set(names)) != len(names) or len({Path(name).stem for name in names}) != len(names):
        raise ValueError(f"Duplicate image names or label stems in {split} manifest")
    image_folder, label_folder = root / "images" / split, root / "labels" / split
    image_paths = [image_folder / name for name in names]
    for row, image_path in zip(selected, image_paths):
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        label = label_folder / (image_path.stem + ".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Missing ground-truth label: {label}")
        boxes = read_boxes(label)
        check_boxes(boxes, label)
        if "visibility" in row and (row["visibility"] not in (0, 1) or len(boxes) != row["visibility"]):
            raise ValueError(f"Manifest visibility and label disagree: {label}")
    # Prepared image folders contain only the manifest's images (hidden files are ignored).
    actual = {path.resolve() for path in image_folder.rglob("*")
              if path.is_file() and not any(part.startswith(".")
                  for part in path.relative_to(image_folder).parts)}
    if actual != {path.resolve() for path in image_paths}:
        raise ValueError(f"Image folder and {split} manifest disagree: {image_folder}")
    actual_labels = {path.resolve() for path in label_folder.rglob("*.txt")
                     if path.is_file() and not any(part.startswith(".")
                         for part in path.relative_to(label_folder).parts)}
    expected_labels = {(label_folder / (path.stem + ".txt")).resolve() for path in image_paths}
    if actual_labels != expected_labels:
        raise ValueError(f"Label folder and {split} manifest disagree: {label_folder}")
    return {"name": dataset_name, "root": root, "split": split,
            "yaml": write_dataset_yaml(config, base, dataset_name),
            "manifest_sha256": sha256(manifest_path), "images": image_paths,
            "labels": label_folder}


def evaluate_checkpoint(weights, model_name, imgsz, data, output, settings):
    # Imported here so --help and file/metric checks do not load the ML library.
    import ultralytics
    from ultralytics import YOLO

    output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = sha256(weights)
    args_path = (weights.parent.parent if weights.parent.name == "weights" else weights.parent) / "args.yaml"
    training_args = None
    if args_path.is_file():
        import yaml
        training_args = yaml.safe_load(args_path.read_text())
    train_imgsz = training_args.get("imgsz") if isinstance(training_args, dict) else None

    model = YOLO(str(weights))
    if model.names != {0: "ball"}:
        raise ValueError(f"Expected one ball class, got {model.names}")
    observed = {}

    def record_validator(validator):
        observed.update(images=int(validator.seen), nms=validator.args.nms,
                        end2end=bool(validator.end2end))

    model.add_callback("on_val_end", record_validator)
    options = dict(data=str(data["yaml"]), split=data["split"], imgsz=imgsz,
                   batch=settings["batch"], device=settings["device"],
                   conf=settings["conf"], iou=settings["iou"], max_det=settings["max_det"],
                   rect=True, quantize=None, augment=False, plots=True,
                   save_txt=True, save_conf=True,
                   project=str(output.parent), name=output.name, exist_ok=True)
    # exist_ok is safe here: this invocation just reserved a brand-new empty directory.
    # None means omit nms, preserving the historical evaluator's default behavior.
    if settings["nms"] is not None:
        options["nms"] = settings["nms"]
    metrics = model.val(**options)
    if observed.get("images") != len(data["images"]):
        raise RuntimeError(f"Validator saw {observed.get('images')} images; manifest has {len(data['images'])}")
    save_dir = Path(metrics.save_dir).resolve()
    if save_dir != output.resolve():
        raise RuntimeError(f"Unexpected validation output directory: {save_dir}")
    mde, frame_rows = calculate_mde(data["images"], data["labels"], save_dir / "labels",
                                    settings["mde_confidence"])
    metric_values = {str(key): float(value) for key, value in metrics.results_dict.items()}
    result = {"schema": "yolo26-evaluation-v1", "model": model_name,
              "train_imgsz": train_imgsz, "eval_imgsz": imgsz,
              "weights": str(weights), "weights_sha256": checkpoint_hash,
              "training_args_file": str(args_path) if args_path.is_file() else None,
              "training_args": training_args, "data": str(data["yaml"]),
              "dataset": data["name"], "dataset_root": str(data["root"]),
              "manifest_sha256": data["manifest_sha256"], "split": data["split"],
              "images": len(data["images"]), "nms": settings["nms"],
              "observed_validator": observed, "settings": {**settings, "rect": True,
                  "quantize": None, "augment": False, "save_txt": True, "save_conf": True},
              "ultralytics": ultralytics.__version__, "mde": mde, "metrics": metric_values,
              "speed_ms_per_image": {str(key): float(value) for key, value in metrics.speed.items()},
              "save_dir": str(save_dir)}
    row = {"model": model_name, "dataset": data["name"], "split": data["split"],
           "train_imgsz": train_imgsz, "eval_imgsz": imgsz,
           "precision": metric_values["metrics/precision(B)"],
           "recall": metric_values["metrics/recall(B)"],
           "mAP50": metric_values["metrics/mAP50(B)"],
           "mAP50_95": metric_values["metrics/mAP50-95(B)"], **mde,
           "inference_ms": result["speed_ms_per_image"].get("inference"),
           "weights": str(weights), "source_json": str(save_dir / "result.json")}
    write_csv(save_dir / "center_errors.csv", frame_rows)
    write_csv(save_dir / "metrics.csv", [row])
    (save_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Results: {save_dir / 'result.json'}")
    print(f"MDE: {mde['mde_px']} pixels; scored={mde['mde_scored_frames']}; misses={mde['missed_positive_frames']}")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", help="Checkpoint alias from project_config.json")
    choice.add_argument("--weights", type=Path, help="Checkpoint path, relative to the config directory")
    choice.add_argument("--matrix", action="store_true", help="Evaluate pilot640/full640/full1280 at 640 and 1280")
    parser.add_argument("--imgsz", type=int, help="Required for a single evaluation; matrix uses 640 and 1280")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--dataset", choices=("full", "pilot"), default="full")
    parser.add_argument("--mde-conf", type=float)
    parser.add_argument("--device", help="Override runtime.device, e.g. 0 or cpu")
    parser.add_argument("--nms", choices=("on", "off", "default"), help="Override the configured setting; default omits the argument")
    args = parser.parse_args(argv)
    if args.matrix and args.imgsz is not None:
        parser.error("--matrix uses both 640 and 1280; omit --imgsz")
    if not args.matrix and (args.imgsz is None or args.imgsz <= 0):
        parser.error("Single evaluation requires a positive --imgsz")
    config, base = load_config(args.config)
    evaluation = config["evaluation"]
    settings = {key: evaluation[key] for key in ("batch", "conf", "iou", "max_det", "mde_confidence")}
    settings["nms"] = evaluation.get("nms")
    settings["device"] = args.device if args.device is not None else config["runtime"]["device"]
    if args.mde_conf is not None:
        settings["mde_confidence"] = args.mde_conf
    if args.nms is not None:
        settings["nms"] = {"on": True, "off": False, "default": None}[args.nms]
    if not 0 <= settings["conf"] <= settings["mde_confidence"] <= 1:
        parser.error("Require 0 <= evaluation.conf <= MDE confidence <= 1")
    if not 0 <= settings["iou"] <= 1 or settings["batch"] <= 0 or settings["max_det"] <= 0:
        parser.error("Invalid evaluation IoU, batch size, or maximum detections")
    if settings["nms"] is not None and not isinstance(settings["nms"], bool):
        parser.error("Configured evaluation.nms must be null, true, or false")

    if args.weights is not None:
        weights = resolve_path(base, args.weights)
        cells = [(weights.stem, weights, args.imgsz)]
    else:
        models = MATRIX_MODELS if args.matrix else (args.model,)
        sizes = MATRIX_SIZES if args.matrix else (args.imgsz,)
        for name in models:
            if name not in config["checkpoints"]:
                parser.error(f"Unknown checkpoint alias: {name}")
        cells = [(name, resolve_path(base, config["checkpoints"][name]), size)
                 for name in models for size in sizes]
    for _, weights, _ in cells:
        if not weights.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {weights}")
    data = prepare_data(config, base, args.dataset, args.split)
    project = resolve_path(base, evaluation["project"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label = "matrix" if args.matrix else f"{cells[0][0]}_eval{args.imgsz}"
    output = project / f"{label}_{args.dataset}_{args.split}_{stamp}"
    if args.matrix:
        output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, weights, size in cells:
        cell_output = output / f"{name}_eval{size}" if args.matrix else output
        rows.append(evaluate_checkpoint(weights, name, size, data, cell_output, settings))
        if args.matrix:
            write_csv(output / "summary.csv", rows)  # Preserve completed cells if a later one fails.
    print(f"Completed {len(rows)}/{len(cells)} evaluations: {output}")


if __name__ == "__main__":
    main()
