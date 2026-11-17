"""
Moved from runModelMatthew.py
"""


import argparse
import csv
import json
import math
from pathlib import Path
import tempfile

from project_config import load_config, resolve_path, dataset_root, write_dataset_yaml


def load_splits(raw_root, dataset):
    """Preserve official membership, optionally selecting match1 through match50."""
    selected = {f"match{i}" for i in range(1, 51)}
    splits, seen = {}, set()
    for split in ("train", "val", "test"):
        entries = json.loads((raw_root / "info" / f"{split}.json").read_text())
        clips = []
        for match, rally in entries:
            if not all(isinstance(x, str) and Path(x).name == x and x not in ("", ".", "..")
                       for x in (match, rally)):
                raise ValueError(f"Invalid clip name in {split}: {(match, rally)}")
            clip = (match, rally)
            if clip in seen:
                raise ValueError(f"Duplicate clip in official splits: {clip}")
            seen.add(clip)
            if dataset == "full" or match in selected:
                clips.append(clip)
        if not clips:
            raise ValueError(f"No selected clips in official {split} split")
        splits[split] = clips
    if dataset == "pilot":
        counts = {split: len(clips) for split, clips in splits.items()}
        matches = {match for clips in splits.values() for match, rally in clips}
        if counts != {"train": 41, "val": 4, "test": 5} or matches != selected:
            raise ValueError("Pilot selection must contain match1–match50 with 41/4/5 clips")
    return splits


def make_label(row, width, height):
    """Convert a visible center to the original clipped 15-pixel YOLO box."""
    visible = int(row["Visibility"])
    if visible not in (0, 1):
        raise ValueError(f"Invalid visibility: {visible}")
    if not visible:
        return "", visible
    x, y = float(row["X"]), float(row["Y"])
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Ball center outside {width}x{height} image: {(x, y)}")
    x1, x2 = max(0, x - 7.5), min(width, x + 7.5)
    y1, y2 = max(0, y - 7.5), min(height, y + 7.5)
    center_x, center_y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
    box_width, box_height = (x2 - x1) / width, (y2 - y1) / height
    return (f"0 {center_x:.9f} {center_y:.9f} "
            f"{box_width:.9f} {box_height:.9f}\n"), visible


def verify_existing(destination, splits):
    """Check membership, file pairs, label structure, and recorded split counts."""
    manifest = json.loads((destination / "manifest.json").read_text())
    summary = json.loads((destination / "summary.json").read_text())
    expected_videos = {split: {f"{match}_{rally}.mp4" for match, rally in clips}
                       for split, clips in splits.items()}
    videos = {split: set() for split in splits}
    images = {split: set() for split in splits}
    counts = {split: {"videos": 0, "images": 0, "positive": 0, "negative": 0}
              for split in splits}
    seen_frames = set()
    for row in manifest:
        split, video, name = row["split"], row["video"], row["image"]
        frame, visible = row["frame"], row["visibility"]
        if split not in splits or video not in expected_videos[split]:
            raise ValueError(f"Manifest sample is outside the selected official splits: {row}")
        if type(frame) is not int or frame < 0 or type(visible) is not int or visible not in (0, 1):
            raise ValueError(f"Invalid manifest frame or visibility: {row}")
        expected_name = f"{Path(video).stem}_frame{frame:06d}.jpg"
        if name != expected_name or (video, frame) in seen_frames or name in images[split]:
            raise ValueError(f"Duplicate or inconsistent manifest sample: {row}")
        seen_frames.add((video, frame))
        videos[split].add(video)
        images[split].add(name)
        image_path = destination / "images" / split / name
        label_path = destination / "labels" / split / (Path(name).stem + ".txt")
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError(f"Missing or empty prepared image: {image_path}")
        lines = label_path.read_text().strip().splitlines()
        if len(lines) != visible:
            raise ValueError(f"Visibility and label row count disagree: {label_path}")
        if visible:
            values = [float(value) for value in lines[0].split()]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Malformed YOLO label: {label_path}")
            cls, x, y, width, height = values
            if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Invalid YOLO class or coordinates: {label_path}")
        counts[split]["images"] += 1
        counts[split]["positive" if visible else "negative"] += 1
    for split in splits:
        if videos[split] != expected_videos[split]:
            raise ValueError(f"Prepared {split} video membership differs from official splits")
        for folder, expected in (("images", images[split]),
                                 ("labels", {Path(name).stem + ".txt" for name in images[split]})):
            directory = destination / folder / split
            actual = {path.name for path in directory.iterdir() if not path.name.startswith(".")}
            if actual != expected:
                raise ValueError(f"Prepared file list differs from manifest: {directory}")
        counts[split]["videos"] = len(videos[split])
        if summary.get(split) != counts[split]:
            raise ValueError(f"Recorded {split} summary disagrees with manifest")
    return counts


def extract_annotated_frames(video, image_directory, annotations):
    """Read frames in order, saving only those listed in the annotation CSV."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    saved_frames = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        # CSV frame numbers start at zero. Stop after the final annotated frame.
        for frame_number in range(max(annotations) + 1):
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Cannot read frame {frame_number}: {video}")
            if frame_number not in annotations:
                continue

            height, width = frame.shape[:2]
            image_name = f"{video.stem}_frame{frame_number:06d}.jpg"
            image_path = image_directory / image_name
            saved = cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not saved:
                raise RuntimeError(f"Cannot save prepared image: {image_path}")
            # Keep names/dimensions for label writing, not the image arrays.
            saved_frames.append({"frame": frame_number, "image": image_name,
                                 "width": width, "height": height})
    finally:
        capture.release()
    if len(saved_frames) != len(annotations):
        raise RuntimeError(f"Not all annotated frames were saved: {video}")
    return saved_frames


def write_frame_labels(saved_frames, annotations, label_directory, video_name, split):
    """Write one matching TXT file per saved image and return its manifest rows."""
    records = []
    for saved in saved_frames:
        frame_number = saved["frame"]
        row = annotations[frame_number]
        label, visible = make_label(row, saved["width"], saved["height"])
        label_name = Path(saved["image"]).stem + ".txt"
        (label_directory / label_name).write_text(label)
        records.append({"split": split, "video": video_name, "frame": frame_number,
                        "visibility": visible, "image": saved["image"]})
    return records


def prepare(raw_root, destination, splits):
    """Save annotated original-size frames and paired labels into a new dataset."""
    if destination.exists():
        raise ValueError(f"Dataset already exists: {destination}; use --reuse to verify it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=destination.name + ".building-", dir=destination.parent))
    manifest, summary = [], {}
    completed = 0
    total = sum(len(clips) for clips in splits.values())
    try:
        for split, clips in splits.items():
            image_directory, label_directory = stage / "images" / split, stage / "labels" / split
            image_directory.mkdir(parents=True)
            label_directory.mkdir(parents=True)
            counts = {"videos": len(clips), "images": 0, "positive": 0, "negative": 0}
            for match, rally in clips:
                annotation = raw_root / "all" / match / "csv" / f"{rally}_ball.csv"
                with annotation.open(newline="") as file:
                    rows = list(csv.DictReader(file))
                targets = {int(row["Frame"]): row for row in rows}
                if not targets or len(targets) != len(rows) or min(targets) < 0:
                    raise ValueError(f"Empty, duplicate, or negative frame indices: {annotation}")
                video = raw_root / "videos" / f"{match}_{rally}.mp4"
                saved_frames = extract_annotated_frames(video, image_directory, targets)
                video_records = write_frame_labels(
                    saved_frames, targets, label_directory, video.name, split)
                manifest.extend(video_records)
                positive = sum(row["visibility"] for row in video_records)
                counts["images"] += len(video_records)
                counts["positive"] += positive
                counts["negative"] += len(video_records) - positive
                completed += 1
                print(f"Prepared {completed}/{total} videos: {video.name}", flush=True)
            summary[split] = counts
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (stage / "README.txt").write_text(
            "Official split membership; annotated frames only; original image dimensions.\n"
            "Zero-based frame indices. Class 0: ball. Visible centers use clipped 15x15 boxes.\n"
            "Visibility 0 produces an empty label. Unannotated frames are excluded.\n")
        verify_existing(stage, splits)
        if destination.exists():
            raise ValueError(f"Destination appeared during preparation: {destination}")
        stage.rename(destination)
        return summary
    except Exception:
        print(f"Preparation stopped. Partial files remain at: {stage}", flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Project JSON; default is beside this script")
    parser.add_argument("--dataset", choices=("full", "pilot"), default="full")
    parser.add_argument("--reuse", action="store_true", help="Verify existing data instead of preparing it")
    args = parser.parse_args(argv)
    config, base = load_config(args.config)
    raw_root = resolve_path(base, config["data"]["raw_root"])
    destination = dataset_root(config, base, args.dataset)
    splits = load_splits(raw_root, args.dataset)
    if args.reuse:
        summary = verify_existing(destination, splits)
        print(f"Verified existing {args.dataset} dataset: {destination}")
    else:
        summary = prepare(raw_root, destination, splits)
        print(f"Prepared {args.dataset} dataset: {destination}")
    yaml_path = write_dataset_yaml(config, base, args.dataset)
    print(json.dumps(summary, indent=2))
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"Error: {error}")
