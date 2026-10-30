#!/usr/bin/env python3
"""Prepare matched single-frame and 3-frame datasets without changing the source.

Each context TIFF contains nine grayscale pages in this order:
previous RGB, center RGB, next RGB. Ultralytics stacks those pages into HWC=9.
The single-frame PNG uses the identical decoded center pixels, losslessly.
Both targets are byte-for-byte copies of the existing center-frame YOLO label.

Rerunning resumes only this script's own incomplete destination. Existing image
or label files are checked, never overwritten. COMPLETE.json records checksums.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image


OFFSETS = (-1, 0, 1)
SCHEMA = "racketvision-context50-v1"
CHANNEL_ORDER = [f"{time}_{color}" for time in ("previous", "center", "next") for color in "RGB"]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def atomic_json(path, value):
    """Replace only our own small progress metadata, never source data."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".metadata-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(json_bytes(value))
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_bytes_if_absent(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Existing file differs; refusing to overwrite: {path}")
        return
    with path.open("xb") as stream:
        stream.write(value)


def save_lossless(path, arrays, multipage=False):
    """Create atomically, or verify pixels left by an interrupted preparation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if multipage:
            ok, existing = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
            valid = ok and len(existing) == len(arrays) and all(
                np.array_equal(a, b) for a, b in zip(existing, arrays)
            )
        else:
            valid = np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), arrays[0])
        if not valid:
            raise ValueError(f"Existing image differs; refusing to overwrite: {path}")
        return
    descriptor, filename = tempfile.mkstemp(dir=path.parent, prefix=".writing-", suffix=path.suffix)
    os.close(descriptor)
    temp = Path(filename)
    try:
        if multipage:
            ok = cv2.imwritemulti(str(temp), arrays, [cv2.IMWRITE_TIFF_COMPRESSION, 5])  # lossless LZW
        else:
            ok = cv2.imwrite(str(temp), arrays[0], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError(f"Cannot encode {path}")
        os.link(temp, path)  # fails rather than replacing another file
    finally:
        temp.unlink(missing_ok=True)


def load_inputs(source, raw):
    manifest_path = source / "manifest.json"
    rows = json.loads(manifest_path.read_text())
    selected = {f"match{i}" for i in range(1, 51)}
    expected, all_seen, split_hashes = {}, set(), {}
    for split in ("train", "val", "test"):
        path = raw / "info" / f"{split}.json"
        entries = json.loads(path.read_text())
        expected[split] = set()
        split_hashes[split] = file_sha256(path)
        for match, rally in entries:
            pair = (match, rally)
            if pair in all_seen:
                raise ValueError(f"Clip repeated across official splits: {pair}")
            all_seen.add(pair)
            if match in selected:
                expected[split].add(f"{match}_{rally}.mp4")
    if {match for match, rally in all_seen if match in selected} != selected:
        raise ValueError("Official splits do not contain every match1 through match50")

    groups, actual, seen_images, seen_frames = defaultdict(list), defaultdict(set), set(), set()
    label_digest = hashlib.sha256()
    source_images = {}
    for row in sorted(rows, key=lambda r: (r["split"], r["video"], int(r["frame"]))):
        split, video, image = row["split"], row["video"], row["image"]
        if split not in expected or video not in expected[split]:
            raise ValueError(f"Sample is outside the official match1–50 split: {row}")
        if Path(image).name != image or Path(video).name != video:
            raise ValueError(f"Expected basename in source manifest: {row}")
        frame, visibility = int(row["frame"]), int(row["visibility"])
        if frame < 0 or visibility not in (0, 1):
            raise ValueError(f"Invalid annotation: {row}")
        key, frame_key = (split, image), (video, frame)
        if key in seen_images or frame_key in seen_frames:
            raise ValueError(f"Duplicate sample: {row}")
        seen_images.add(key)
        seen_frames.add(frame_key)
        stem = Path(image).stem
        if stem != f"{Path(video).stem}_frame{frame:06d}":
            raise ValueError(f"Filename and frame index disagree: {row}")
        image_path, label_path = source / "images" / split / image, source / "labels" / split / f"{stem}.txt"
        label = label_path.read_bytes()
        lines = label.decode().strip().splitlines()
        if len(lines) != visibility:
            raise ValueError(f"Visibility/label mismatch: {label_path}")
        if visibility:
            values = [float(s) for s in lines[0].split()]
            if len(values) != 5 or not all(math.isfinite(x) for x in values):
                raise ValueError(f"Malformed YOLO label: {label_path}")
            cls, x, y, width, height = values
            if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Invalid YOLO class/coordinates: {label_path}")
        with Image.open(image_path) as im:
            shape = [im.height, im.width]
        stat = image_path.stat()
        source_images[f"{split}/{image}"] = [stat.st_size, stat.st_mtime_ns, shape]
        label_digest.update(f"{split}/{stem}.txt\0".encode() + label + b"\0")
        groups[(split, video)].append(dict(row, frame=frame, visibility=visibility, source_shape=shape))
        actual[split].add(video)
    if dict(actual) != expected:
        raise ValueError("Source manifest video membership differs from the official match1–50 subsets")

    videos = {}
    for split, video in groups:
        path = raw / "videos" / video
        stat = path.stat()
        videos[video] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    inputs = {
        "source_yolo": str(source), "raw_root": str(raw), "source_manifest_sha256": file_sha256(manifest_path),
        "official_split_sha256": split_hashes, "source_labels_sha256": label_digest.hexdigest(),
        "source_image_metadata": source_images, "source_video_metadata": videos,
        "offsets": list(OFFSETS), "channels": 9, "channel_order": CHANNEL_ORDER,
    }
    # Video content is not rehashed: size/mtime changes invalidate resumption.
    fingerprint = hashlib.sha256(json_bytes(inputs)).hexdigest()
    return groups, inputs, fingerprint


def write_sample(destination, source, row, frames, indices, frame_count, fps):
    split, stem = row["split"], Path(row["image"]).stem
    center = frames[1]
    if list(center.shape[:2]) != row["source_shape"]:
        raise ValueError(f"Decoded/source image dimensions disagree: {row['image']}")
    if any(f.dtype != np.uint8 or f.shape != center.shape or f.shape[2] != 3 for f in frames):
        raise ValueError(f"Expected three equally sized uint8 BGR frames: {row['image']}")
    # Nine 2-D pages, NOT three HWC pages: native Ultralytics stacks page axis=2.
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    pages = [np.ascontiguousarray(frame[:, :, channel]) for frame in rgb for channel in range(3)]
    context = Path("images") / split / f"{stem}.tiff"
    single = Path("single/images") / split / f"{stem}.png"
    label = Path("labels") / split / f"{stem}.txt"
    single_label = Path("single/labels") / split / f"{stem}.txt"
    save_lossless(destination / context, pages, multipage=True)
    save_lossless(destination / single, [center])
    label_bytes = (source / "labels" / split / f"{stem}.txt").read_bytes()
    for relative in (label, single_label):
        write_bytes_if_absent(destination / relative, label_bytes)
    requested = [row["frame"] + offset for offset in OFFSETS]
    return {
        "split": split, "video": row["video"], "frame": row["frame"], "visibility": row["visibility"],
        "source_image": row["image"], "shape_hw": list(center.shape[:2]), "video_frames": frame_count, "fps": fps,
        "offsets": list(OFFSETS), "requested_indices": requested, "actual_indices": indices,
        "clamped_context_frames": sum(a != b for a, b in zip(requested, indices)),
        "image": str(context), "single_image": str(single), "label": str(label), "single_label": str(single_label),
        "sha256": {str(p): file_sha256(destination / p) for p in (context, single, label, single_label)},
    }


def prepare_video(destination, source, raw, rows):
    """Sequentially decode with a tiny rolling cache; never hold a whole clip."""
    path = raw / "videos" / rows[0]["video"]
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        count, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
        if count <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video metadata: {path}")
        ready, last_use = defaultdict(list), {}
        for row in rows:
            if row["frame"] >= count:
                raise ValueError(f"Annotated frame is outside video: {row}")
            indices = [min(max(row["frame"] + offset, 0), count - 1) for offset in OFFSETS]
            end = max(indices)
            ready[end].append((row, indices))
            for index in indices:
                last_use[index] = max(last_use.get(index, -1), end)
        cache, records = {}, []
        for index in range(max(ready) + 1):
            if not capture.grab():
                raise RuntimeError(f"Cannot decode frame {index}: {path}")
            if index in last_use:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"Cannot retrieve frame {index}: {path}")
                cache[index] = frame
            for row, indices in ready.get(index, []):
                records.append(write_sample(destination, source, row, [cache[i] for i in indices], indices, count, fps))
            for cached in list(cache):
                if last_use[cached] <= index:
                    del cache[cached]
        return records
    finally:
        capture.release()


def verify_records(destination, records):
    seen = set()
    for record in records:
        key = (record["split"], record["video"], record["frame"])
        if key in seen:
            raise ValueError(f"Duplicate output manifest record: {key}")
        seen.add(key)
        for relative, expected in record["sha256"].items():
            path = destination / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"Invalid artifact path in manifest: {relative}")
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"Missing/changed artifact; refusing to overwrite: {path}")


def yaml_bytes(path, channels):
    return f"path: {json.dumps(str(path))}\ntrain: images/train\nval: images/val\ntest: images/test\nchannels: {channels}\nnames:\n  0: ball\n".encode()


def run(source, raw, destination, verify_only=False):
    source, raw, destination = source.resolve(), raw.resolve(), destination.resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Destination must be separate from the source dataset")
    groups, inputs, fingerprint = load_inputs(source, raw)
    state_path, manifest_path, complete_path = destination / "summary.json", destination / "manifest.json", destination / "COMPLETE.json"
    if destination.exists():
        if not state_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"Destination is not a recognized resumable dataset: {destination}")
        state = json.loads(state_path.read_text())
        if state.get("schema") != SCHEMA or state.get("input_fingerprint") != fingerprint:
            raise ValueError("Source inputs/configuration changed; refusing to reuse this destination")
        records = json.loads(manifest_path.read_text())
        verify_records(destination, records)
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            for relative, digest in complete["sha256"].items():
                if file_sha256(destination / relative) != digest:
                    raise ValueError(f"Completion checksum mismatch: {relative}")
            if len(records) != sum(map(len, groups.values())):
                raise ValueError("Completed dataset has the wrong sample count")
            print(f"Already complete; all output checksums verified: {destination}", flush=True)
            return
        if verify_only:
            raise ValueError("Destination is incomplete; existing artifact checksums passed")
    else:
        if verify_only:
            raise ValueError(f"Destination does not exist: {destination}")
        destination.mkdir(parents=True)
        state = {"schema": SCHEMA, "state": "preparing", "input_fingerprint": fingerprint, "inputs": inputs}
        records = []
        atomic_json(manifest_path, records)
        atomic_json(state_path, state)

    write_bytes_if_absent(destination / "context.yaml", yaml_bytes(destination, 9))
    write_bytes_if_absent(destination / "single.yaml", yaml_bytes(destination / "single", 3))
    done = defaultdict(set)
    for record in records:
        done[(record["split"], record["video"])].add(record["frame"])
    if set(done) - set(groups):
        raise ValueError("Output manifest contains an unexpected video")
    for position, ((split, video), rows) in enumerate(sorted(groups.items()), 1):
        frames = {row["frame"] for row in rows}
        if (split, video) in done:
            if done[(split, video)] != frames:
                raise ValueError(f"Unexpected partial video manifest: {video}")
            print(f"Verified {position}/{len(groups)}: {video} ({len(rows)} samples)", flush=True)
            continue
        records.extend(prepare_video(destination, source, raw, rows))
        atomic_json(manifest_path, records)
        print(f"Prepared {position}/{len(groups)}: {video} ({len(rows)} samples)", flush=True)

    split_summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        split_summary[split] = {
            "videos": len({r["video"] for r in subset}), "images": len(subset),
            "positive": sum(r["visibility"] for r in subset), "negative": sum(not r["visibility"] for r in subset),
            "samples_with_boundary_clamping": sum(r["clamped_context_frames"] > 0 for r in subset),
            "clamped_context_frames": sum(r["clamped_context_frames"] for r in subset),
            "shapes_hw": dict(Counter(f"{r['shape_hw'][0]}x{r['shape_hw'][1]}" for r in subset)),
        }
    if len(records) != sum(map(len, groups.values())):
        raise ValueError("Output sample count differs from the source manifest")
    state.update(state="complete", splits=split_summary, total_samples=len(records))
    atomic_json(state_path, state)
    atomic_json(complete_path, {
        "schema": SCHEMA, "input_fingerprint": fingerprint, "samples": len(records),
        "sha256": {name: file_sha256(destination / name) for name in ("manifest.json", "summary.json", "context.yaml", "single.yaml")},
    })
    print(json.dumps(split_summary, indent=2), flush=True)
    print(f"Complete: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-yolo", type=Path, default=Path("/home/gonga2/data/racketvision50_yolo"))
    parser.add_argument("--raw-root", type=Path, default=Path("/home/gonga2/data/RacketVision/tennis"))
    parser.add_argument("--destination", type=Path, default=Path("/home/gonga2/data/racketvision50_context3"))
    parser.add_argument("--verify-only", action="store_true", help="Verify completion and all artifact checksums without writing")
    args = parser.parse_args()
    run(args.source_yolo, args.raw_root, args.destination, args.verify_only)


if __name__ == "__main__":
    main()
