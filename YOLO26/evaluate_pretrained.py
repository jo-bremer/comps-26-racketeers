# AI generated to test data validity. Preliminary steps before actual model training
"""Compare pretrained YOLO detections before/after the current stationary filter.

Copy this file into the server's YOLO26 directory and run it there.
Only official validation videos within match1-match50 are evaluated.
No training, test evaluation, or edits to runModel.py / existing CSVs.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
import ultralytics

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/gonga2/data/racketvision50_yolo')
SOURCE = Path('/home/gonga2/data/RacketVision/tennis')
CONF = 0.2
IMGSZ = 640
RADIUS = 1.0  # Original-image pixels, matching runModel.py.
MATCH_IOU = 0.5  # Evaluation criterion, independent of NMS IoU.


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def stationary_locations(frames):
    # Reproduce runModel.py: first 10 frames, excluding empty detections.
    centers = [[center(box) for box, _ in frame]
               for frame in frames[:10] if frame]
    candidates = []
    for previous, current in zip(centers, centers[1:]):
        for point in current:
            if any(math.dist(point, other) <= RADIUS for other in previous):
                candidates.append(point)

    # runModel.py then compares each candidate with itself; all pass.
    # Omitting that no-op preserves its effective behavior.
    anchors, groups = [], []
    for point in candidates:
        group_id = None
        for index, anchor in enumerate(anchors):
            if math.dist(point, anchor) <= RADIUS:
                group_id = index  # Original code uses the last matching group.
        if group_id is None:
            anchors.append(point)
            groups.append([])
            group_id = len(groups) - 1
        groups[group_id].append(point)

    return [(sum(p[0] for p in group) / len(group),
             sum(p[1] for p in group) / len(group)) for group in groups]


def apply_filter(frame, locations):
    return [(box, confidence) for box, confidence in frame
            if not any(math.dist(center(box), location) <= RADIUS
                       for location in locations)]


def box_iou(a, b):
    intersection = (max(0, min(a[2], b[2]) - max(a[0], b[0])) *
                    max(0, min(a[3], b[3]) - max(a[1], b[1])))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def ground_truth(row, shape):
    label = DATA / 'labels/val' / (Path(row['image']).stem + '.txt')
    lines = label.read_text().splitlines()
    if row['visibility'] == 0:
        assert not lines, f'Unexpected positive label: {label}'
        return None
    assert row['visibility'] == 1 and len(lines) == 1, label
    cls, x, y, w, h = map(float, lines[0].split())
    assert cls == 0, label
    height, width = shape
    return [(x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height]


def counts(frame, gt):
    # One annotated target per positive frame: at most one TP.
    tp = int(gt is not None and any(box_iou(box, gt) >= MATCH_IOU
                                    for box, _ in frame))
    return {'TP': tp, 'FP': len(frame) - tp,
            'FN': int(gt is not None) - tp,
            'frames': 1, 'candidates': len(frame),
            'multi_candidate_frames': int(len(frame) > 1),
            'negative_frames': int(gt is None),
            'negative_frames_with_predictions': int(gt is None and bool(frame))}


def add(total, values):
    for key, value in values.items():
        total[key] = total.get(key, 0) + value


def metrics(total):
    tp, fp, fn = total['TP'], total['FP'], total['FN']
    return {**total, 'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'negative_frame_false_positive_rate': (
                total['negative_frames_with_predictions'] / total['negative_frames']
                if total['negative_frames'] else None),
            'candidates_per_frame': total['candidates'] / total['frames']}


def save_predictions(path, frames):
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame_index_0', 'boxes_xyxy', 'confidences'])
        for index, frame in enumerate(frames):
            writer.writerow([index, json.dumps([x[0] for x in frame]),
                             json.dumps([x[1] for x in frame])])


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    selected = {f'match{i}_000.mp4' for i in range(1, 51)}
    assert {row['video'] for row in manifest} == selected
    official = {f'{match}_{clip}.mp4' for match, clip in
                json.loads((SOURCE / 'info/val.json').read_text())}
    rows = [row for row in manifest if row['split'] == 'val']
    videos = sorted({row['video'] for row in rows})
    assert set(videos) == selected & official, 'Validation split mismatch'
    assert len(rows) == 200 and sum(r['visibility'] for r in rows) == 193

    model = YOLO(str(ROOT / 'yolo26x.pt'))
    assert model.names.get(32) == 'sports ball', 'Use original COCO weights'
    out = ROOT / 'runs/val' / datetime.now().strftime('stationary_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    totals = {'raw': {}, 'filtered': {}}
    per_video, frame_details = {}, []
    lost_targets = 0

    for video in videos:
        print(f'Processing {video} ...', flush=True)
        raw, shape = [], None
        # Predict ALL consecutive video frames once, with runModel settings.
        for result in model.predict(source=str(SOURCE / 'videos' / video),
                                    conf=CONF, classes=[32], imgsz=IMGSZ,
                                    iou=0.7, max_det=300, rect=True, vid_stride=1,
                                    device=0, stream=True, save=False, verbose=False):
            if shape is None:
                shape = result.orig_shape
            assert shape == result.orig_shape, 'Unexpected frame size change'
            raw.append(list(zip(result.boxes.xyxy.cpu().tolist(),
                                result.boxes.conf.cpu().tolist())))

        locations = stationary_locations(raw)
        filtered = [apply_filter(frame, locations) for frame in raw]
        video_totals = {'raw': {}, 'filtered': {}}
        video_lost = 0
        annotated = [row for row in rows if row['video'] == video]
        assert len({row['frame'] for row in annotated}) == len(annotated)
        for row in annotated:
            index = row['frame']  # Official frame indices start at ZERO.
            assert 0 <= index < len(raw), f'Missing annotated frame: {row}'
            gt = ground_truth(row, shape)
            before, after = counts(raw[index], gt), counts(filtered[index], gt)
            lost = before['TP'] - after['TP']
            assert lost >= 0  # A removal-only filter cannot create a new match.
            video_lost += lost
            for branch, value in [('raw', before), ('filtered', after)]:
                add(totals[branch], value)
                add(video_totals[branch], value)
            frame_details.append({'video': video, 'frame_index_0': index,
                                  'visibility': row['visibility'],
                                  'raw_candidates': len(raw[index]),
                                  'filtered_candidates': len(filtered[index]),
                                  'raw_TP': before['TP'], 'filtered_TP': after['TP'],
                                  'target_lost': lost})

        lost_targets += video_lost
        per_video[video] = {k: metrics(v) for k, v in video_totals.items()}
        per_video[video].update(stationary_locations=locations,
                                target_frames_lost=video_lost,
                                decoded_frames=len(raw))
        save_predictions(out / f'{Path(video).stem}_raw.csv', raw)
        save_predictions(out / f'{Path(video).stem}_filtered.csv', filtered)
        print(f'{video}: {len(raw)} frames; {len(locations)} stationary locations; '
              f'{video_lost} annotated target frames lost')

    summary = {'config': {'weights': str(ROOT / 'yolo26x.pt'),
                         'ultralytics': ultralytics.__version__, 'split': 'val',
                         'videos': videos, 'imgsz': IMGSZ, 'conf': CONF,
                         'stationary_radius_pixels': RADIUS,
                         'stationary_context_frames': 10, 'match_iou': MATCH_IOU,
                         'scored_frames': 'annotated frames only'},
               'raw': metrics(totals['raw']), 'filtered': metrics(totals['filtered']),
               'removed_false_positives': totals['raw']['FP'] - totals['filtered']['FP'],
               'target_frames_lost': lost_targets, 'per_video': per_video}
    (out / 'comparison.json').write_text(json.dumps(summary, indent=2))
    with (out / 'frame_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_details[0]))
        writer.writeheader()
        writer.writerows(frame_details)
    print('\nFixed conf=0.2, matching IoU>=0.5; NOT mAP / best-F1 metrics')
    for branch in ['raw', 'filtered']:
        m = summary[branch]
        print(f"{branch:8s}: TP={m['TP']} FP={m['FP']} FN={m['FN']} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    print('Removed false positives:', summary['removed_false_positives'])
    print('Previously detected target frames lost:', lost_targets)
    print('Results saved to:', out)


if __name__ == '__main__':
    main()