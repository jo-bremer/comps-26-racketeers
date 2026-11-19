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
    """Read YOLO label rows as lists of numbers."""
    if not path.exists():
        return []  # A missing prediction file means no detections.

    boxes = []
    text = path.read_text()
    for line in text.splitlines():
        if not line.strip():
            continue

        box = []
        for field in line.split():
            number = float(field)
            box.append(number)
        boxes.append(box)

    return boxes


def check_boxes(boxes, path, predictions=False):
    """Check the number, class, coordinates, and confidence of each box."""
    if not predictions and len(boxes) > 1:
        raise ValueError(f"Expected at most one labeled ball: {path}")

    fields = 5
    if predictions:
        fields = 6

    for box in boxes:
        valid_fields = len(box) == fields
        if valid_fields:
            for value in box:
                if not isfinite(value):
                    valid_fields = False
                    break
        if not valid_fields:
            raise ValueError(f"Expected {fields} finite YOLO fields: {path}")

        valid_coordinates = box[0] == 0
        if valid_coordinates:
            for value in box[1:5]:
                if value < 0 or value > 1:
                    valid_coordinates = False
                    break
        if not valid_coordinates:
            raise ValueError(f"Expected class 0 and normalized coordinates: {path}")

        if not predictions:
            if box[3] <= 0 or box[4] <= 0:
                raise ValueError(f"Ground-truth box dimensions must be positive: {path}")

        # Clipping may give a prediction zero area; it still counts for MDE.
        if predictions:
            if box[5] < 0 or box[5] > 1:
                raise ValueError(f"Prediction confidence must be between 0 and 1: {path}")


def calculate_mde(image_paths, label_folder, prediction_folder, confidence):
    """Measure center errors in original pixels and count missed balls."""
    from PIL import Image

    if not 0 <= confidence <= 1:
        raise ValueError("MDE confidence must be between 0 and 1")

    errors = []
    frame_rows = []
    positive_frames = 0
    missed_frames = 0
    negative_false_positives = 0

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

        # Keep the first box when two predictions have the same confidence.
        prediction = None
        for box in predictions:
            if box[5] >= confidence:
                if prediction is None:
                    prediction = box
                elif box[5] > prediction[5]:
                    prediction = box

        with Image.open(image_path) as image:
            width, height = image.size

        gt_x = None
        gt_y = None
        pred_x = None
        pred_y = None
        distance = None
        prediction_confidence = None

        if prediction is not None:
            pred_x = prediction[1] * width
            pred_y = prediction[2] * height
            prediction_confidence = prediction[5]

        if ground_truth:
            positive_frames += 1
            gt_x = ground_truth[0][1] * width
            gt_y = ground_truth[0][2] * height
            if prediction is None:
                missed_frames += 1
            else:
                x_error = pred_x - gt_x
                y_error = pred_y - gt_y
                distance = hypot(x_error, y_error)
                errors.append(distance)
        elif prediction is not None:
            negative_false_positives += 1

        frame_row = {
            "image": image_path.name,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "distance_px": distance,
            "prediction_confidence": prediction_confidence,
        }
        frame_rows.append(frame_row)

    mde_px = None
    if errors:
        mde_px = sum(errors) / len(errors)

    prediction_coverage = None
    if positive_frames:
        prediction_coverage = len(errors) / positive_frames

    mde = {
        "mde_confidence": confidence,
        "mde_px": mde_px,
        "mde_scored_frames": len(errors),
        "positive_frames": positive_frames,
        "missed_positive_frames": missed_frames,
        "prediction_coverage": prediction_coverage,
        "negative_frames_with_prediction": negative_false_positives,
    }
    return mde, frame_rows


def sha256(path):
    """Calculate a file's SHA-256 hash without reading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        block = file.read(1024 * 1024)
        while block:
            digest.update(block)
            block = file.read(1024 * 1024)
    return digest.hexdigest()


def write_csv(path, rows):
    """Write rows using the first row's keys as the CSV columns."""
    with path.open("w", newline="") as file:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def visible_files(folder, pattern):
    """Find matching files, skipping hidden files and hidden folders."""
    files = set()
    for path in folder.rglob(pattern):
        if not path.is_file():
            continue

        relative_path = path.relative_to(folder)
        hidden = False
        for part in relative_path.parts:
            if part.startswith("."):
                hidden = True
                break

        if not hidden:
            files.add(path.resolve())

    return files


def prepare_data(config, base, dataset_name, split):
    """Check the manifest, images, and labels before loading a model."""
    root = dataset_root(config, base, dataset_name)
    manifest_path = root / "manifest.json"
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)

    selected = []
    for row in manifest:
        if row["split"] == split:
            selected.append(row)
    if not selected:
        raise ValueError(f"No {split} images in {manifest_path}")

    names = []
    for row in selected:
        names.append(row["image"])

    for name in names:
        if not isinstance(name, str):
            raise ValueError("Manifest image names must be plain filenames")
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Manifest image names must be plain filenames")

    unique_names = set(names)
    unique_stems = set()
    for name in names:
        unique_stems.add(Path(name).stem)
    if len(unique_names) != len(names) or len(unique_stems) != len(names):
        raise ValueError(f"Duplicate image names or label stems in {split} manifest")

    image_folder = root / "images" / split
    label_folder = root / "labels" / split
    image_paths = []
    for name in names:
        image_paths.append(image_folder / name)

    for row, image_path in zip(selected, image_paths):
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        label_name = image_path.stem + ".txt"
        label = label_folder / label_name
        if not label.is_file():
            raise FileNotFoundError(f"Missing ground-truth label: {label}")

        boxes = read_boxes(label)
        check_boxes(boxes, label)
        if "visibility" in row:
            visibility = row["visibility"]
            if visibility not in (0, 1) or len(boxes) != visibility:
                raise ValueError(f"Manifest visibility and label disagree: {label}")

    # Visible files must match the manifest exactly.
    actual_images = visible_files(image_folder, "*")
    expected_images = set()
    for image_path in image_paths:
        expected_images.add(image_path.resolve())
    if actual_images != expected_images:
        raise ValueError(f"Image folder and {split} manifest disagree: {image_folder}")

    actual_labels = visible_files(label_folder, "*.txt")
    expected_labels = set()
    for image_path in image_paths:
        label_name = image_path.stem + ".txt"
        label_path = label_folder / label_name
        expected_labels.add(label_path.resolve())
    if actual_labels != expected_labels:
        raise ValueError(f"Label folder and {split} manifest disagree: {label_folder}")

    data = {
        "name": dataset_name,
        "root": root,
        "split": split,
        "yaml": write_dataset_yaml(config, base, dataset_name),
        "manifest_sha256": sha256(manifest_path),
        "images": image_paths,
        "labels": label_folder,
    }
    return data


def evaluate_checkpoint(weights, model_name, imgsz, data, output, settings):
    """Validate one checkpoint and save its metrics and center errors."""
    # Keep this import here so --help and data checks do not load the ML library.
    import ultralytics
    from ultralytics import YOLO

    output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = sha256(weights)

    args_folder = weights.parent
    if weights.parent.name == "weights":
        args_folder = weights.parent.parent
    args_path = args_folder / "args.yaml"

    training_args = None
    if args_path.is_file():
        import yaml
        training_args = yaml.safe_load(args_path.read_text())

    train_imgsz = None
    if isinstance(training_args, dict):
        train_imgsz = training_args.get("imgsz")

    model = YOLO(str(weights))
    if model.names != {0: "ball"}:
        raise ValueError(f"Expected one ball class, got {model.names}")

    observed = {}

    def record_validator(validator):
        """Save the image count and settings reported by the validator."""
        observed["images"] = int(validator.seen)
        observed["nms"] = validator.args.nms
        observed["end2end"] = bool(validator.end2end)

    model.add_callback("on_val_end", record_validator)
    options = {
        "data": str(data["yaml"]),
        "split": data["split"],
        "imgsz": imgsz,
        "batch": settings["batch"],
        "device": settings["device"],
        "conf": settings["conf"],
        "iou": settings["iou"],
        "max_det": settings["max_det"],
        "rect": True,
        "quantize": None,
        "augment": False,
        "plots": True,
        "save_txt": True,
        "save_conf": True,
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
    }
    # This run just created an empty output folder, so exist_ok is safe.
    # Leave nms out when it is None to keep the library's default behavior.
    if settings["nms"] is not None:
        options["nms"] = settings["nms"]

    metrics = model.val(**options)
    observed_images = observed.get("images")
    expected_images = len(data["images"])
    if observed_images != expected_images:
        raise RuntimeError(
            f"Validator saw {observed_images} images; manifest has {expected_images}"
        )

    save_dir = Path(metrics.save_dir).resolve()
    if save_dir != output.resolve():
        raise RuntimeError(f"Unexpected validation output directory: {save_dir}")

    prediction_folder = save_dir / "labels"
    mde, frame_rows = calculate_mde(
        data["images"], data["labels"], prediction_folder, settings["mde_confidence"]
    )

    metric_values = {}
    for key, value in metrics.results_dict.items():
        metric_values[str(key)] = float(value)

    training_args_file = None
    if args_path.is_file():
        training_args_file = str(args_path)

    saved_settings = settings.copy()
    saved_settings["rect"] = True
    saved_settings["quantize"] = None
    saved_settings["augment"] = False
    saved_settings["save_txt"] = True
    saved_settings["save_conf"] = True

    speed_ms_per_image = {}
    for key, value in metrics.speed.items():
        speed_ms_per_image[str(key)] = float(value)

    result = {
        "schema": "yolo26-evaluation-v1",
        "model": model_name,
        "train_imgsz": train_imgsz,
        "eval_imgsz": imgsz,
        "weights": str(weights),
        "weights_sha256": checkpoint_hash,
        "training_args_file": training_args_file,
        "training_args": training_args,
        "data": str(data["yaml"]),
        "dataset": data["name"],
        "dataset_root": str(data["root"]),
        "manifest_sha256": data["manifest_sha256"],
        "split": data["split"],
        "images": len(data["images"]),
        "nms": settings["nms"],
        "observed_validator": observed,
        "settings": saved_settings,
        "ultralytics": ultralytics.__version__,
        "mde": mde,
        "metrics": metric_values,
        "speed_ms_per_image": speed_ms_per_image,
        "save_dir": str(save_dir),
    }

    row = {
        "model": model_name,
        "dataset": data["name"],
        "split": data["split"],
        "train_imgsz": train_imgsz,
        "eval_imgsz": imgsz,
        "precision": metric_values["metrics/precision(B)"],
        "recall": metric_values["metrics/recall(B)"],
        "mAP50": metric_values["metrics/mAP50(B)"],
        "mAP50_95": metric_values["metrics/mAP50-95(B)"],
    }
    for key, value in mde.items():
        row[key] = value
    row["inference_ms"] = result["speed_ms_per_image"].get("inference")
    row["weights"] = str(weights)
    row["source_json"] = str(save_dir / "result.json")

    write_csv(save_dir / "center_errors.csv", frame_rows)
    write_csv(save_dir / "metrics.csv", [row])
    result_path = save_dir / "result.json"
    result_text = json.dumps(result, indent=2) + "\n"
    result_path.write_text(result_text)

    print(f"Results: {save_dir / 'result.json'}")
    print(
        f"MDE: {mde['mde_px']} pixels; "
        f"scored={mde['mde_scored_frames']}; "
        f"misses={mde['missed_positive_frames']}"
    )
    return row


def main(argv=None):
    """Read command-line options and run a single evaluation or the matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", help="Checkpoint alias from project_config.json")
    choice.add_argument(
        "--weights", type=Path,
        help="Checkpoint path, relative to the config directory"
    )
    choice.add_argument(
        "--matrix", action="store_true",
        help="Evaluate pilot640/full640/full1280 at 640 and 1280"
    )
    parser.add_argument(
        "--imgsz", type=int,
        help="Required for a single evaluation; matrix uses 640 and 1280"
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--dataset", choices=("full", "pilot"), default="full")
    parser.add_argument("--mde-conf", type=float)
    parser.add_argument("--device", help="Override runtime.device, e.g. 0 or cpu")
    parser.add_argument(
        "--nms", choices=("on", "off", "default"),
        help="Override the configured setting; default omits the argument"
    )
    args = parser.parse_args(argv)

    if args.matrix and args.imgsz is not None:
        parser.error("--matrix uses both 640 and 1280; omit --imgsz")
    if not args.matrix:
        if args.imgsz is None or args.imgsz <= 0:
            parser.error("Single evaluation requires a positive --imgsz")

    config, base = load_config(args.config)
    evaluation = config["evaluation"]
    settings = {}
    setting_names = ("batch", "conf", "iou", "max_det", "mde_confidence")
    for key in setting_names:
        settings[key] = evaluation[key]
    settings["nms"] = evaluation.get("nms")

    if args.device is not None:
        settings["device"] = args.device
    else:
        settings["device"] = config["runtime"]["device"]
    if args.mde_conf is not None:
        settings["mde_confidence"] = args.mde_conf
    if args.nms == "on":
        settings["nms"] = True
    elif args.nms == "off":
        settings["nms"] = False
    elif args.nms == "default":
        settings["nms"] = None

    if not 0 <= settings["conf"] <= settings["mde_confidence"] <= 1:
        parser.error("Require 0 <= evaluation.conf <= MDE confidence <= 1")
    if not 0 <= settings["iou"] <= 1:
        parser.error("Invalid evaluation IoU, batch size, or maximum detections")
    if settings["batch"] <= 0 or settings["max_det"] <= 0:
        parser.error("Invalid evaluation IoU, batch size, or maximum detections")
    if settings["nms"] is not None:
        if not isinstance(settings["nms"], bool):
            parser.error("Configured evaluation.nms must be null, true, or false")

    cells = []
    if args.weights is not None:
        weights = resolve_path(base, args.weights)
        cells.append((weights.stem, weights, args.imgsz))
    else:
        if args.matrix:
            models = MATRIX_MODELS
            sizes = MATRIX_SIZES
        else:
            models = (args.model,)
            sizes = (args.imgsz,)

        for name in models:
            if name not in config["checkpoints"]:
                parser.error(f"Unknown checkpoint alias: {name}")

        for name in models:
            for size in sizes:
                weights = resolve_path(base, config["checkpoints"][name])
                cells.append((name, weights, size))

    for name, weights, size in cells:
        if not weights.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {weights}")

    data = prepare_data(config, base, args.dataset, args.split)
    project = resolve_path(base, evaluation["project"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.matrix:
        label = "matrix"
    else:
        label = f"{cells[0][0]}_eval{args.imgsz}"
    output = project / f"{label}_{args.dataset}_{args.split}_{stamp}"
    if args.matrix:
        output.mkdir(parents=True, exist_ok=False)

    rows = []
    for name, weights, size in cells:
        if args.matrix:
            cell_output = output / f"{name}_eval{size}"
        else:
            cell_output = output

        row = evaluate_checkpoint(weights, name, size, data, cell_output, settings)
        rows.append(row)
        if args.matrix:
            # Keep completed cells even if a later evaluation fails.
            write_csv(output / "summary.csv", rows)

    print(f"Completed {len(rows)}/{len(cells)} evaluations: {output}")


if __name__ == "__main__":
    main()
