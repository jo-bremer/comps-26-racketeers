"""
AI generated for testing post processing procedure, we decided not to spend more time on csv postprocessing
removing the extra ball on the side of the court.
Compare raw and stationary-filtered predictions on the historical pilot validation set.

Default protocol: first 50 videos, official val intersection, 200 annotated frames
(193 positive). Fixed-confidence P/R/F1 are NOT model.val best-F1 metrics or mAP.
Relative paths use the config folder. No training or test evaluation is performed.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from project_config import dataset_root, load_config, resolve_path
from runModel import apply_filter, predict_video, resolve_ball_class, stationary_locations, validate_settings


def box_iou(a, b):
    intersection = (max(0, min(a[2], b[2]) - max(a[0], b[0])) *
                    max(0, min(a[3], b[3]) - max(a[1], b[1])))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def ground_truth(row, shape, data):
    label = data / "labels/val" / (Path(row["image"]).stem + ".txt")
    lines = label.read_text().splitlines()
    if row["visibility"] == 0:
        if lines:
            raise ValueError(f"Unexpected positive label: {label}")
        return None
    if row["visibility"] != 1 or len(lines) != 1:
        raise ValueError(f"Expected one positive ball label: {label}")
    cls, x, y, w, h = map(float, lines[0].split())
    if (cls != 0 or not all(math.isfinite(value) for value in (cls, x, y, w, h))
            or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)):
        raise ValueError(f"Expected a finite class-0 box with normalized coordinates: {label}")
    height, width = shape
    if not all(math.isfinite(value) and value > 0 for value in (height, width)):
        raise ValueError(f"Invalid original image dimensions: {shape}")
    return [(x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height]


def counts(frame, gt, match_iou=0.5):
    # One labeled target per positive frame: at most one true-positive box.
    tp = int(gt is not None and any(box_iou(box, gt) >= match_iou for box, _ in frame))
    return {"TP": tp, "FP": len(frame) - tp, "FN": int(gt is not None) - tp,
            "frames": 1, "candidates": len(frame),
            "multi_candidate_frames": int(len(frame) > 1),
            "negative_frames": int(gt is None),
            "negative_frames_with_predictions": int(gt is None and bool(frame))}


def add(total, values):
    for key, value in values.items():
        total[key] = total.get(key, 0) + value


def metrics(total):
    tp, fp, fn = total["TP"], total["FP"], total["FN"]
    return {**total, "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "negative_frame_false_positive_rate": (
                total["negative_frames_with_predictions"] / total["negative_frames"]
                if total["negative_frames"] else None),
            "candidates_per_frame": total["candidates"] / total["frames"]}


def save_predictions(path, frames):
    # Keep this evaluator's original ZERO-based schema for annotation alignment.
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index_0", "boxes_xyxy", "confidences"])
        for index, frame in enumerate(frames):
            writer.writerow([index, json.dumps([box for box, _ in frame]),
                             json.dumps([confidence for _, confidence in frame])])


def validation_rows(data, source, first_matches=50):
    manifest = json.loads((data / "manifest.json").read_text())
    selected = {f"match{i}_000.mp4" for i in range(1, first_matches + 1)}
    if {row["video"] for row in manifest} != selected:
        raise ValueError("Pilot manifest does not contain exactly the requested first-match videos.")
    official = {f"{match}_{clip}.mp4" for match, clip in
                json.loads((source / "info/val.json").read_text())}
    rows = [row for row in manifest if row["split"] == "val"]
    videos = sorted({row["video"] for row in rows})
    if set(videos) != selected & official or not rows:
        raise ValueError("Pilot validation split does not match the official split or is empty.")
    if first_matches == 50 and (len(rows) != 200 or sum(r["visibility"] for r in rows) != 193):
        raise ValueError("Historical first-50 protocol requires 200 val frames, including 193 positives.")
    return rows, videos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--first-matches", type=int, default=50,
                        help="Pilot manifest must contain exactly match1..matchN, clip 000")
    parser.add_argument("--weights", help="Default: model.pretrained; an explicit custom ball checkpoint is allowed")
    parser.add_argument("--class-id", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    if args.first_matches <= 0:
        parser.error("--first-matches must be positive")
    cfg, base = load_config(args.config)
    settings = cfg["stationary_evaluation"]
    data = dataset_root(cfg, base, "pilot")
    source = resolve_path(base, cfg["data"]["raw_root"])
    weights = resolve_path(base, args.weights or cfg["model"]["pretrained"])
    conf = args.conf if args.conf is not None else settings["conf"]
    imgsz = args.imgsz if args.imgsz is not None else settings["imgsz"]
    device = args.device if args.device is not None else cfg.get("runtime", {}).get("device", 0)
    radius, context_frames, match_iou = settings["radius"], settings["frames"], settings["match_iou"]
    validate_settings(conf, imgsz, radius, context_frames)
    if not math.isfinite(match_iou) or not 0 < match_iou <= 1:
        raise ValueError("Matching IoU must be in (0, 1].")
    rows, videos = validation_rows(data, source, args.first_matches)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    for video in videos:
        if not (source / "videos" / video).is_file():
            raise FileNotFoundError(source / "videos" / video)

    from ultralytics import YOLO
    import ultralytics
    model = YOLO(str(weights))
    class_id = resolve_ball_class(model.names, args.class_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = resolve_path(base, settings["project"]) / stamp
    out.mkdir(parents=True, exist_ok=False)
    totals = {"raw": {}, "filtered": {}}
    per_video, frame_details, lost_targets = {}, [], 0
    for video in videos:
        print(f"Processing {video} (class {class_id}) ...", flush=True)
        raw, shape, _ = predict_video(model, source / "videos" / video,
                                      class_id=class_id, conf=conf, imgsz=imgsz, device=device)
        save_predictions(out / f"{Path(video).stem}_raw.csv", raw)
        locations = stationary_locations(raw, radius, context_frames)
        filtered = [apply_filter(frame, locations, radius) for frame in raw]
        save_predictions(out / f"{Path(video).stem}_filtered.csv", filtered)
        video_totals, video_lost = {"raw": {}, "filtered": {}}, 0
        annotated = [row for row in rows if row["video"] == video]
        if len({row["frame"] for row in annotated}) != len(annotated):
            raise ValueError(f"Duplicate annotated frame index in {video}")
        for row in annotated:
            index = row["frame"]  # Official annotations start at ZERO.
            if not isinstance(index, int) or not 0 <= index < len(raw):
                raise ValueError(f"Missing annotated frame: {row}")
            gt = ground_truth(row, shape, data)
            before, after = counts(raw[index], gt, match_iou), counts(filtered[index], gt, match_iou)
            lost = before["TP"] - after["TP"]
            if lost < 0:
                raise ValueError("A removal-only filter unexpectedly created a new match.")
            video_lost += lost
            for branch, value in (("raw", before), ("filtered", after)):
                add(totals[branch], value)
                add(video_totals[branch], value)
            frame_details.append({"video": video, "frame_index_0": index,
                                  "visibility": row["visibility"], "raw_candidates": len(raw[index]),
                                  "filtered_candidates": len(filtered[index]),
                                  "raw_TP": before["TP"], "filtered_TP": after["TP"],
                                  "target_lost": lost})
        lost_targets += video_lost
        per_video[video] = {key: metrics(value) for key, value in video_totals.items()}
        per_video[video].update(stationary_locations=locations, target_frames_lost=video_lost,
                                decoded_frames=len(raw))

    summary = {"config": {"weights": str(weights), "model_names": model.names, "class_id": class_id,
                           "ultralytics": ultralytics.__version__, "split": "val",
                           "dataset": str(data), "source": str(source), "first_matches": args.first_matches,
                           "videos": videos, "imgsz": imgsz, "conf": conf, "device": device,
                           "stationary_radius_pixels": radius, "stationary_context_frames": context_frames,
                           "match_iou": match_iou, "frame_index_base": 0,
                           "nms_argument": "omitted", "iou_argument": 0.7, "max_det": 300,
                           "rect": True, "vid_stride": 1, "scored_frames": "annotated frames only",
                           "metrics_protocol": "fixed-confidence P/R/F1; not mAP or best-F1 metrics"},
               "raw": metrics(totals["raw"]), "filtered": metrics(totals["filtered"]),
               "removed_false_positives": totals["raw"]["FP"] - totals["filtered"]["FP"],
               "target_frames_lost": lost_targets, "per_video": per_video}
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "frame_comparison.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_details[0]))
        writer.writeheader()
        writer.writerows(frame_details)
    print(f"Fixed conf={conf:g}, matching IoU>={match_iou:g}; NOT mAP / best-F1 metrics")
    for branch in ("raw", "filtered"):
        m = summary[branch]
        print(f"{branch:8s}: TP={m['TP']} FP={m['FP']} FN={m['FN']} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    print(f"Previously detected target frames lost: {lost_targets}")
    print(f"Results saved to: {out}")
    return out


if __name__ == "__main__":
    main()
