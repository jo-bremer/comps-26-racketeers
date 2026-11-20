"""Run YOLO on videos and save raw and filtered detections as CSV files. Filtered stationary detections"""
import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from project_config import load_config, resolve_path


def center(box):
    """Find the middle (x, y) point of a box stored as [x1, y1, x2, y2]."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def stationary_locations(frames, radius=1.0, context_frames=10):
    """Estimate stationary positions from start of a video.

   The returned positions are used to filter detections later
    """
    centers = [[center(box) for box, _ in frame]
               for frame in frames[:context_frames] if frame]
    candidates = []
    for previous, current in zip(centers, centers[1:]):
        for point in current:
            if any(math.dist(point, other) <= radius for other in previous):
                candidates.append(point)

    anchors, groups = [], []
    for point in candidates:
        group_id = None
        for index, anchor in enumerate(anchors):
            if math.dist(point, anchor) <= radius:
                group_id = index  
        if group_id is None:
            anchors.append(point)
            groups.append([])
            group_id = len(groups) - 1
        groups[group_id].append(point)
    return [(sum(p[0] for p in group) / len(group),
             sum(p[1] for p in group) / len(group)) for group in groups]


def apply_filter(frame, locations, radius=1.0):
    """
    Return a new list so the raw detections are still available for comparison.
    """
    return [(box, confidence) for box, confidence in frame
            if not any(math.dist(center(box), location) <= radius
                       for location in locations)]


def resolve_ball_class(names, class_id=None):
    """Find the ball's class number, or use the class number given by the user."""
    names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
    names = {int(index): str(name) for index, name in names.items()}
    if class_id is not None:
        if class_id not in names:
            raise ValueError(f"Class {class_id} is absent from model.names: {names}")
        return class_id
    recognized = {"ball", "tennis ball", "sports ball"}
    choices = [index for index, name in names.items()
               if name.lower().replace("_", " ").strip() in recognized]
    return choices[0] 


def video_sort_key(path):
    """Sort match videos by their numbers, so match2_0 comes before match10_0."""
    match = re.fullmatch(r"match(\d+)_(\d+)", path.stem)
    # Put other filenames after the match videos and sort them alphabetically.
    return (0, int(match[1]), int(match[2])) if match else (1, path.name)


def find_videos(source):
    """Get one video file, or the sorted .mp4 files directly inside a folder."""
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Video source does not exist: {source}")
    videos = sorted(source.glob("*.mp4"), key=video_sort_key)
    if not videos:
        raise ValueError(f"No .mp4 videos found in {source}")
    return videos


def predict_video(model, video, *, class_id, conf, imgsz, device,
                  save_video=False, video_project=None):
    """Run the model on a video and collect boxes and confidence scores per frame.

    Return the detections, frame size (height, width), and saved-video folder.
    The folder is None when saving videos is turned off.
    """
    # stream=True gives us one result at a time; vid_stride=1 uses every frame.
    options = dict(source=str(video), conf=conf, classes=[class_id], imgsz=imgsz,
                   iou=0.7, max_det=300, rect=True, vid_stride=1,
                   device=device, stream=True, save=save_video, verbose=False)
    if video_project is not None:
        options.update(project=str(video_project), name=video.stem, exist_ok=False)
    frames, shape, saved_video_dir = [], None, None
    for result in model.predict(**options):
        # Record the first frame's size and check that later frames have the same size.
        current_shape = tuple(result.orig_shape)
        if shape is None:
            shape = current_shape
        if current_shape != shape:
            raise ValueError(f"Video frame size changed in {video}")
        # Keep only detection data, not the full images. An empty list keeps a frame
        # with no detections in the correct position for the CSV output.
        frames.append(list(zip(result.boxes.xyxy.cpu().tolist(),
                               result.boxes.conf.cpu().tolist())))
        if save_video:
            # These saved videos show raw predictions, before our stationary filter.
            saved_video_dir = str(result.save_dir)
    if not frames:
        raise ValueError(f"No video frames were decoded: {video}")
    return frames, shape, saved_video_dir


def frame_summary(frames):
    """Count frames and detections for the summary. These counts are not accuracy."""
    # A frame counts as detected if it has at least one box.
    detected = sum(bool(frame) for frame in frames)
    return {"frames": len(frames), "frames_with_detection": detected,
            "detection_rate": detected / len(frames) if frames else None,
            "boxes": sum(len(frame) for frame in frames),
            "frames_with_multiple_detections": sum(len(frame) > 1 for frame in frames)}


def save_legacy_csv(path, frames):
    """Save one row per frame using the project's existing CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mode "x" creates a new file and refuses to overwrite an existing file.
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_number", "visibility", "xyxy", "confidence"])
        # Frame numbers start at 1. Here, visibility means a prediction was found;
        # it is not a ground-truth label. Empty frames have blank box/confidence cells.
        for index, frame in enumerate(frames, start=1):
            boxes = [box for box, _ in frame]
            confidences = [confidence for _, confidence in frame]
            writer.writerow([index, int(bool(frame)), boxes if frame else "",
                             confidences if frame else ""])
        stats = frame_summary(frames)
        # Keep the old six-field summary row after the four-column frame rows.
        writer.writerow(["Detection count is", stats["frames_with_detection"],
                         "the ratio is", stats["detection_rate"],
                         "total frames", stats["frames"]])


def validate_settings(conf, imgsz, radius, context_frames):
    """Check that confidence, image size, and stationary-filter settings are valid."""
    if not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    if not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer.")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Stationary radius must be finite and nonnegative.")
    if not isinstance(context_frames, int) or context_frames <= 0:
        raise ValueError("Stationary context frames must be a positive integer.")


def main(argv=None):
    """Read settings, load the model, and use the helper functions on each video."""
    # Read command-line options, such as the source video and model weights.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", help="One video or an .mp4 folder; default: raw_root/videos")
    parser.add_argument("--weights", help="Default: model.pretrained from the config")
    parser.add_argument("--class-id", type=int, help="Override class selection from model.names")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--device", help="Ultralytics device, for example 0 or cpu")
    parser.add_argument("--save-video", action="store_true",
                        help="Also save videos with RAW model boxes; the temporal filter changes CSVs only")
    args = parser.parse_args(argv)
    # Use command-line values when supplied; otherwise use the config defaults.
    cfg, base = load_config(args.config)
    settings = cfg["inference"]
    weights = resolve_path(base, args.weights or cfg["model"]["pretrained"])
    source = (resolve_path(base, args.source) if args.source else
              resolve_path(base, cfg["data"]["raw_root"]) / "videos")
    conf = args.conf if args.conf is not None else settings["conf"]
    imgsz = args.imgsz if args.imgsz is not None else settings["imgsz"]
    device = args.device if args.device is not None else cfg.get("runtime", {}).get("device", 0)
    radius, context_frames = settings["stationary_radius"], settings["stationary_frames"]
    # Check the inputs before loading the model or starting video processing.
    validate_settings(conf, imgsz, radius, context_frames)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    videos = find_videos(source)

    # Load the model once and reuse it for every video.
    from ultralytics import YOLO
    import ultralytics
    model = YOLO(str(weights))
    class_id = resolve_ball_class(model.names, args.class_id)
    # Give this run a new output folder and record its settings in summary.json.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = resolve_path(base, settings["project"]) / stamp
    out.mkdir(parents=True, exist_ok=False)
    record = {"state": "running", "weights": str(weights), "model_names": model.names,
              "class_id": class_id, "source": str(source), "ultralytics": ultralytics.__version__,
              "conf": conf, "imgsz": imgsz, "device": device, "frame_index_base": 1,
              "stationary_radius_pixels": radius, "stationary_context_frames": context_frames,
              "nms_argument": "omitted", "iou_argument": 0.7, "max_det": 300,
              "rect": True, "vid_stride": 1, "save_video": args.save_video,
              "video_annotations": "raw predictions, before the stationary filter", "videos": []}
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        for video in videos:
            print(f"Processing {video.name} (class {class_id}) ...", flush=True)
            raw, shape, saved_dir = predict_video(
                model, video, class_id=class_id, conf=conf, imgsz=imgsz, device=device,
                save_video=args.save_video, video_project=out / "videos" if args.save_video else None)
            # Save the original detections first so we can compare them with the filter.
            raw_csv, filtered_csv = out / "raw" / f"{video.stem}.csv", out / "filtered" / f"{video.stem}.csv"
            save_legacy_csv(raw_csv, raw)
            # Estimate locations once from this video's opening frames, then apply
            # those same locations to every frame in the video.
            locations = stationary_locations(raw, radius, context_frames)
            filtered = [apply_filter(frame, locations, radius) for frame in raw]
            save_legacy_csv(filtered_csv, filtered)
            # Update the summary after each video so completed results are recorded.
            record["videos"].append({"source": str(video), "shape_hw": shape,
                                     "raw": frame_summary(raw), "filtered": frame_summary(filtered),
                                     "stationary_locations": locations, "raw_csv": str(raw_csv),
                                     "filtered_csv": str(filtered_csv), "annotated_video_dir": saved_dir})
            summary_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"Saved raw and filtered CSVs for {len(raw)} frames.")
    except Exception as exc:
        # Record the problem and re-raise it so the run still reports an error.
        record.update(state="failed", error=str(exc))
        summary_path.write_text(json.dumps(record, indent=2) + "\n")
        raise
    # All videos finished successfully.
    record["state"] = "completed"
    summary_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Outputs: {out}")
    return out


if __name__ == "__main__":
    # Run the program when this file is executed directly, not when it is imported.
    main()
