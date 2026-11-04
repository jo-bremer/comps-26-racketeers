import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import cv2


# Keep new run logs separate from the original 50-video job.
JOB = Path.home() / 'comps-training-jobs' / (
    'ball_full_train_' + time.strftime('%Y%m%d-%H%M%S')
)

# Define the source dataset, output dataset, and training script.
SOURCE = Path.home() / 'data/RacketVision/tennis'
DEST = Path.home() / 'data/racketvision_yolo'
SCRIPT = Path.home() / 'comps-26-racketeers/YOLO26/trainModel_finetune.py'


def status(state, **extra):
    """Save and print the current job status."""

    # Add the current time and any extra information.
    record = {
        'state': state,
        'updated_at': time.time(),
        **extra,
    }

    # Write to a temporary file before replacing the status file.
    temporary_file = JOB / 'status.tmp'
    temporary_file.write_text(json.dumps(record, indent=2))
    temporary_file.replace(JOB / 'status.json')

    # Print immediately so the latest status appears in the console.
    print(json.dumps(record), flush=True)


def load_splits():
    """Read every clip in the official train, val, and test split files."""
    split_clips = {}
    seen = set()

    for split in ('train', 'val', 'test'):
        split_file = SOURCE / 'info' / f'{split}.json'
        entries = json.loads(split_file.read_text())
        clips = []

        for match, rally in entries:
            clip = (match, rally)
            assert clip not in seen, f'Duplicate clip in official splits: {clip}'
            seen.add(clip)
            clips.append(clip)

        assert clips, f'Empty official split: {split_file}'
        split_clips[split] = clips

    return split_clips


def prepare(split_clips):
    """Create a YOLO dataset from all officially split annotated frames."""

    # Stop if the output dataset already exists.
    assert not DEST.exists(), (
        f'Dataset already exists: {DEST}; refusing to overwrite'
    )

    # Build the dataset in a temporary directory first.
    stage_name = DEST.name + '.building-' + JOB.name
    stage = DEST.with_name(stage_name)
    stage.mkdir(parents=True, exist_ok=False)

    total_videos = sum(len(clips) for clips in split_clips.values())

    # Track processed clips, dataset counts, and image records.
    seen = set()
    summary = {}
    manifest = []

    # Preserve every clip's membership in its official split.
    for split, clips in split_clips.items():

        # Create separate directories for images and labels.
        image_directory = stage / 'images' / split
        label_directory = stage / 'labels' / split

        image_directory.mkdir(parents=True)
        label_directory.mkdir(parents=True)

        image_count = 0
        positive_count = 0
        negative_count = 0

        for match, rally in clips:
            # A clip must not appear more than once across the splits.
            clip = (match, rally)
            assert clip not in seen
            seen.add(clip)

            # Read the ball annotations for this clip.
            label_path = (
                SOURCE / 'all' / match / 'csv' / f'{rally}_ball.csv'
            )

            with label_path.open(newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)

            # Map each frame index to its annotation row.
            targets = {}

            for row in rows:
                frame_number = int(row['Frame'])
                targets[frame_number] = row

            # Require at least one annotation and no duplicate frame indices.
            assert targets and len(targets) == len(rows)

            # Frame indices must be zero or greater.
            assert min(targets) >= 0

            video = SOURCE / 'videos' / f'{match}_{rally}.mp4'
            capture = cv2.VideoCapture(str(video))

            assert capture.isOpened(), video

            found = 0

            try:
                last_frame_index = max(targets)

                # Read frames from index 0 through the last annotated frame.
                for frame_index in range(last_frame_index + 1):
                    grabbed = capture.grab()

                    assert grabbed, (
                        f'Cannot decode frame {frame_index}: {video}'
                    )

                    # Skip frames that do not have annotations.
                    if frame_index not in targets:
                        continue

                    retrieved, frame = capture.retrieve()

                    assert retrieved, (
                        f'Cannot retrieve frame {frame_index}: {video}'
                    )

                    height, width = frame.shape[:2]

                    row = targets[frame_index]
                    visible = int(row['Visibility'])

                    assert visible in (0, 1)

                    # An invisible ball produces an empty label file.
                    label = ''

                    if visible == 1:
                        x = float(row['X'])
                        y = float(row['Y'])

                        # The annotated center must be inside the image.
                        assert (
                            0 <= x < width and 0 <= y < height
                        ), (video, frame_index, x, y)

                        # Create a 15-by-15 box around the ball center.
                        # Clip the box to the image boundaries.
                        x1 = max(0, x - 7.5)
                        x2 = min(width, x + 7.5)
                        y1 = max(0, y - 7.5)
                        y2 = min(height, y + 7.5)

                        # Convert the box to normalized YOLO coordinates.
                        center_x = (x1 + x2) / (2 * width)
                        center_y = (y1 + y2) / (2 * height)
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height

                        # Class 0 represents the ball.
                        label = (
                            f'0 {center_x:.9f} {center_y:.9f} '
                            f'{box_width:.9f} {box_height:.9f}\n'
                        )

                        positive_count += 1
                    else:
                        negative_count += 1

                    # Pad the frame index with zeros to at least six digits.
                    name = f'{match}_{rally}_frame{frame_index:06d}'

                    image_path = image_directory / (name + '.jpg')
                    label_file = label_directory / (name + '.txt')

                    # Save the image with JPEG quality set to 95.
                    assert cv2.imwrite(
                        str(image_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )

                    label_file.write_text(label)

                    # Record which video and frame produced this image.
                    manifest.append({
                        'split': split,
                        'video': video.name,
                        'frame': frame_index,
                        'visibility': visible,
                        'image': name + '.jpg',
                    })

                    image_count += 1
                    found += 1

            finally:
                # Release the video even if an error occurs.
                capture.release()

            # Every annotated frame must have been saved.
            assert found == len(targets)

            status(
                'preparing',
                completed_videos=len(seen),
                total_videos=total_videos,
                current_video=video.name,
            )

        # Save the counts for this split.
        summary[split] = {
            'videos': len(clips),
            'images': image_count,
            'positive': positive_count,
            'negative': negative_count,
        }

    # Check that every official clip was processed exactly once.
    expected_clips = {
        clip for clips in split_clips.values() for clip in clips
    }
    assert seen == expected_clips and len(seen) == total_videos

    # Verify the number of saved images and label files.
    for split in summary:
        image_files = list(
            (stage / 'images' / split).glob('*.jpg')
        )
        label_files = list(
            (stage / 'labels' / split).glob('*.txt')
        )

        expected_images = summary[split]['images']

        assert len(image_files) == expected_images
        assert len(label_files) == expected_images

    # Write dataset metadata.
    manifest_file = stage / 'manifest.json'
    summary_file = stage / 'summary.json'
    readme_file = stage / 'README.txt'

    manifest_file.write_text(json.dumps(manifest, indent=2))
    summary_file.write_text(json.dumps(summary, indent=2))

    readme_file.write_text(
        'All clips from official tennis/info/train.json, val.json, and test.json; annotated frames only.\n'
        'Frame indices are zero-based. Class 0: ball. Visible centers get clipped 15x15 boxes.\n'
        'Visibility 0 produces an empty label. Unannotated frames are excluded.\n'
    )

    # Move the completed dataset to its final location.
    stage.rename(DEST)

    return summary


def active_csv_jobs():
    """Return active runModel.py process IDs owned by the current user."""

    jobs = []

    # Linux exposes process information through /proc.
    for process_directory in Path('/proc').iterdir():
        # Process directories have numeric names.
        if not process_directory.name.isdigit():
            continue

        try:
            # Ignore processes owned by other users.
            process_owner = process_directory.stat().st_uid

            if process_owner != os.getuid():
                continue

            # Command-line arguments are separated by null bytes.
            command_file = process_directory / 'cmdline'
            arguments = command_file.read_bytes().split(b'\0')

            # Check whether any argument names runModel.py.
            runs_model_script = False

            for argument in arguments:
                if not argument:
                    continue

                argument_text = argument.decode(errors='replace')
                argument_name = Path(argument_text).name

                if argument_name == 'runModel.py':
                    runs_model_script = True
                    break

            if not runs_model_script:
                continue

            # The process state follows the final ")" in the stat file.
            stat_file = process_directory / 'stat'
            stat_text = stat_file.read_text()
            fields_after_name = stat_text.rsplit(')', 1)[1].split()
            process_state = fields_after_name[0]

            # Ignore stopped, traced, zombie, and dead processes.
            ignored_states = ('T', 't', 'Z', 'X')

            if process_state not in ignored_states:
                process_id = int(process_directory.name)
                jobs.append(process_id)

        except (OSError, ProcessLookupError):
            # A process may exit while its information is being read.
            continue

    return sorted(jobs)


def main():
    """Prepare the dataset, wait for existing jobs, and start training."""

    JOB.mkdir(parents=True, exist_ok=False)
    split_clips = load_splits()
    total_videos = sum(len(clips) for clips in split_clips.values())

    # Save the original script hash and a snapshot of the training script.
    script_hash = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()

    snapshot_file = JOB / 'trainModel.snapshot.py'
    snapshot_file.write_bytes(SCRIPT.read_bytes())

    status(
        'preparing',
        completed_videos=0,
        total_videos=total_videos,
    )

    summary = prepare(split_clips)

    status(
        'dataset_ready',
        dataset=str(DEST),
        summary=summary,
    )

    # Wait until there are no active runModel.py jobs for this user.
    previous_jobs = None

    while True:
        jobs = active_csv_jobs()

        if not jobs:
            break

        # Update the status only when the waiting list changes.
        if jobs != previous_jobs:
            status(
                'queued',
                waiting_for_csv_pids=jobs,
                dataset=str(DEST),
                summary=summary,
            )
            previous_jobs = jobs

        time.sleep(30)

    # Stop if the training script changed during preparation or waiting.
    current_script_hash = hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()

    assert current_script_hash == script_hash, (
        'trainModel_finetune.py changed while queued; review before restarting'
    )

    log_path = JOB / 'train.log'

    with log_path.open('w') as log:
        # Use the current Python interpreter.
        # The -u option disables stdout and stderr buffering.
        command = [
            sys.executable,
            '-u',
            str(SCRIPT),
        ]

        # Send both normal output and error output to the log file.
        child = subprocess.Popen(
            command,
            cwd=SCRIPT.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        status(
            'training',
            pid=child.pid,
            dataset=str(DEST),
            summary=summary,
        )

        # Wait for training to finish and collect its exit code.
        result = child.wait()

    if result != 0:
        raise RuntimeError(
            f'trainModel_finetune.py exited with {result}; see {JOB / "train.log"}'
        )

    status(
        'complete',
        exit_code=0,
        dataset=str(DEST),
        summary=summary,
        train_log=str(log_path),
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Print the full traceback and record the failure.
        traceback.print_exc()
        status('failed', error=str(exc))
        sys.exit(1)