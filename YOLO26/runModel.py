"""Run ball inference; post process csv
"""
import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from project_config import load_config, resolve_path


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def stationary_locations(frames, radius=1.0, context_frames=10):
    """Preserve the original first-window heuristic, including skipped empty frames."""
    centers = [[center(box) for box, _ in frame]
               for frame in frames[:context_frames] if frame]
    candidates = []
    for previous, current in zip(centers, centers[1:]):
        for point in current:
            if any(math.dist(point, other) <= radius for other in previous):
                candidates.append(point)

    # The original candidate-versus-itself test always passed; it was a no-op.
    anchors, groups = [], []
    for point in candidates:
        group_id = None
        for index, anchor in enumerate(anchors):
            if math.dist(point, anchor) <= radius:
                group_id = index  # Preserve the LAST matching group, not the first.
        if group_id is None:
            anchors.append(point)
            groups.append([])
            group_id = len(groups) - 1
        groups[group_id].append(point)
    return [(sum(p[0] for p in group) / len(group),
             sum(p[1] for p in group) / len(group)) for group in groups]


def apply_filter(frame, locations, radius=1.0):
    """Remove boxes near learned locations; leave the raw frame list unchanged."""
    return [(box, confidence) for box, confidence in frame
            if not any(math.dist(center(box), location) <= radius
                       for location in locations)]


def resolve_ball_class(names, class_id=None):
    """Use model metadata, never the checkpoint filename, to choose a ball class."""
    names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
    names = {int(index): str(name) for index, name in names.items()}
    if class_id is not None:
        if class_id not in names:
            raise ValueError(f"Class {class_id} is absent from model.names: {names}")
        return class_id
    recognized = {"ball", "tennis ball", "sports ball"}
    choices = [index for index, name in names.items()
               if name.lower().replace("_", " ").strip() in recognized]
    if len(choices) != 1:
        raise ValueError(f"Cannot identify one ball class in {names}; supply --class-id.")
    return choices[0]  # COCO sports ball is 32; the project's custom ball is 0.


def video_sort_key(path):
    match = re.fullmatch(r"match(\d+)_(\d+)", path.stem)
    return (0, int(match[1]), int(match[2])) if match else (1, path.name)


def find_videos(source):
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
    """Consume the generator, retaining boxes/confidences rather than image arrays."""
    options = dict(source=str(video), conf=conf, classes=[class_id], imgsz=imgsz,
                   iou=0.7, max_det=300, rect=True, vid_stride=1,
                   device=device, stream=True, save=save_video, verbose=False)
    if video_project is not None:
        options.update(project=str(video_project), name=video.stem, exist_ok=False)
    frames, shape, saved_video_dir = [], None, None
    for result in model.predict(**options):
        current_shape = tuple(result.orig_shape)
        if shape is None:
            shape = current_shape
        if current_shape != shape:
            raise ValueError(f"Video frame size changed in {video}")
        frames.append(list(zip(result.boxes.xyxy.cpu().tolist(),
                               result.boxes.conf.cpu().tolist())))
        if save_video:
            saved_video_dir = str(result.save_dir)
    if not frames:
        raise ValueError(f"No video frames were decoded: {video}")
    return frames, shape, saved_video_dir


def frame_summary(frames):
    detected = sum(bool(frame) for frame in frames)
    return {"frames": len(frames), "frames_with_detection": detected,
            "detection_rate": detected / len(frames) if frames else None,
            "boxes": sum(len(frame) for frame in frames),
            "frames_with_multiple_detections": sum(len(frame) > 1 for frame in frames)}


def save_legacy_csv(path, frames):
    """Keep the historical four-column, one-based format and summary footer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_number", "visibility", "xyxy", "confidence"])
        for index, frame in enumerate(frames, start=1):
            boxes = [box for box, _ in frame]
            confidences = [confidence for _, confidence in frame]
            writer.writerow([index, int(bool(frame)), boxes if frame else "",
                             confidences if frame else ""])
        stats = frame_summary(frames)
        writer.writerow(["Detection count is", stats["frames_with_detection"],
                         "the ratio is", stats["detection_rate"],
                         "total frames", stats["frames"]])


def validate_settings(conf, imgsz, radius, context_frames):
    if not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    if not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer.")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Stationary radius must be finite and nonnegative.")
    if not isinstance(context_frames, int) or context_frames <= 0:
        raise ValueError("Stationary context frames must be a positive integer.")


def main(argv=None):
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
    cfg, base = load_config(args.config)
    settings = cfg["inference"]
    weights = resolve_path(base, args.weights or cfg["model"]["pretrained"])
    source = (resolve_path(base, args.source) if args.source else
              resolve_path(base, cfg["data"]["raw_root"]) / "videos")
    conf = args.conf if args.conf is not None else settings["conf"]
    imgsz = args.imgsz if args.imgsz is not None else settings["imgsz"]
    device = args.device if args.device is not None else cfg.get("runtime", {}).get("device", 0)
    radius, context_frames = settings["stationary_radius"], settings["stationary_frames"]
    validate_settings(conf, imgsz, radius, context_frames)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    videos = find_videos(source)

    from ultralytics import YOLO
    import ultralytics
    model = YOLO(str(weights))
    class_id = resolve_ball_class(model.names, args.class_id)
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
            raw_csv, filtered_csv = out / "raw" / f"{video.stem}.csv", out / "filtered" / f"{video.stem}.csv"
            save_legacy_csv(raw_csv, raw)
            locations = stationary_locations(raw, radius, context_frames)
            filtered = [apply_filter(frame, locations, radius) for frame in raw]
            save_legacy_csv(filtered_csv, filtered)
            record["videos"].append({"source": str(video), "shape_hw": shape,
                                     "raw": frame_summary(raw), "filtered": frame_summary(filtered),
                                     "stationary_locations": locations, "raw_csv": str(raw_csv),
                                     "filtered_csv": str(filtered_csv), "annotated_video_dir": saved_dir})
            summary_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"Saved raw and filtered CSVs for {len(raw)} frames.")
    except Exception as exc:
        record.update(state="failed", error=str(exc))
        summary_path.write_text(json.dumps(record, indent=2) + "\n")
        raise
    record["state"] = "completed"
    summary_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Outputs: {out}")
    return out


if __name__ == "__main__":
    main()
