"""Test a saved tennis ball model on every image in the full test split.

Use the path to  model .pt file after --weights.
Use the image size you want after --imgsz.
The script saves your scores in metrics.csv in a new folder under runs/evaluation/.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from math import hypot, isfinite
from pathlib import Path

from project_config import load_config, resolve_path, dataset_root, write_dataset_yaml


def read_boxes(path):
    """Read YOLO label rows as lists of numbers."""
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
        raise ValueError(f"Expected at most one labeled ball")
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


def calculate_mde(image_paths, label_folder, prediction_folder, confidence):
    """Measure center errors in original pixels and count missed balls."""
    from PIL import Image

    errors = []
    frame_rows = []
    positive_frames = 0
    missed_frames = 0
    negative_false_positives = 0

    for image_path in image_paths:
        filename = image_path.stem + ".txt"
        label_path = label_folder / filename
        ground_truth = read_boxes(label_path)
        prediction_path = prediction_folder / filename
        predictions = read_boxes(prediction_path) if prediction_path.is_file() else []
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
    """This is recommended by LLM to increase efficiency.
    Calculate a file's SHA-256 hash without reading all parts. Make it faster
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        #read the first block, at most 1MB
        block = file.read(1024 * 1024)
        #process if this block has data
        while block:
            digest.update(block)
            #read the next package, get 0 byte at the end of file
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
    """Check images and labels before loading a model."""
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
    unique_stems = set()
    for name in names:
        unique_stems.add(Path(name).stem)
    image_folder = root / "images" / split
    label_folder = root / "labels" / split
    image_paths = []
    for name in names:
        image_paths.append(image_folder / name)
    for row, image_path in zip(selected, image_paths):

        label_name = image_path.stem + ".txt"
        label = label_folder / label_name

        boxes = read_boxes(label)
        check_boxes(boxes, label)
        if "visibility" in row:
            visibility = row["visibility"]
            if visibility not in (0, 1) or len(boxes) != visibility:
                raise ValueError(f"Manifest visibility and label disagree: {label}")

    # Visible files must match exactly.
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
    # Leave nms out when it is None to keep the lib's default behavior.
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
    """Read the checkpoint and image size and run one evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    args = parser.parse_args(argv)
    if args.imgsz <= 0:
        parser.error("Single evaluation requires a positive --imgsz")

    config, base = load_config()
    evaluation = config["evaluation"]
    settings = {
        "batch": evaluation["batch"],
        "conf": evaluation["conf"],
        "iou": evaluation["iou"],
        "max_det": evaluation["max_det"],
        "mde_confidence": evaluation["mde_confidence"],
        "nms": evaluation.get("nms"),
        "device": config["runtime"]["device"],
    }

    # Keep every prediction that could qualify for MDE.
    if not 0 <= settings["conf"] <= settings["mde_confidence"] <= 1:
        parser.error("Expected 0 <= conf <= mde_confidence <= 1")

    weights = resolve_path(base, args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {weights}")

    data = prepare_data(config, base, "full", "test")
    project = resolve_path(base, evaluation["project"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = project / f"{weights.stem}_eval{args.imgsz}_full_test_{stamp}"
    evaluate_checkpoint(weights, weights.stem, args.imgsz, data, output, settings)


if __name__ == "__main__":
    main()
