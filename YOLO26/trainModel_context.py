"""Compare three-frame ball detection with a matched single-frame control."""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.patches import imread

ROOT = Path(__file__).resolve().parent
DATA = Path("/home/gonga2/data/racketvision50_context3")
SOURCE = ROOT / "runs/train/ball50_x_640_adamw_30ep/weights/best.pt"


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def center_batch(batch):
    if batch["img"].shape[1] == 9:
        return {**batch, "img": batch["img"][:, 3:6]}
    return batch


class ContextValidator(DetectionValidator):
    """Display the target frame, while evaluating all nine input channels."""

    def plot_val_samples(self, batch, ni):
        super().plot_val_samples(center_batch(batch), ni)

    def plot_predictions(self, batch, preds, ni, max_det=None):
        super().plot_predictions(center_batch(batch), preds, ni, max_det)


class ContextTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(copy.deepcopy(cfg), weights, verbose)
        if weights is not None:
            old = weights.model[0].conv.weight
            new = model.model[0].conv.weight
            if new.shape[1] == 9 and old.shape[1] == 3:
                with torch.no_grad():
                    new.zero_()
                    new[:, 3:6].copy_(old)
                model.context_offsets = [-1, 0, 1]
                model.target_channels = [3, 6]
                print("Initialized center-frame channels from the single-frame checkpoint; context channels start at zero.")
        return model

    def get_validator(self):
        return ContextValidator(self.test_loader, save_dir=self.save_dir,
                                args=copy.copy(self.args), _callbacks=self.callbacks)

    def plot_training_samples(self, batch, ni):
        super().plot_training_samples(center_batch(batch), ni)


def configuration(args, arm):
    return dict(
        model=str(SOURCE), data=str(DATA / ("context.yaml" if arm == "context" else "single.yaml")),
        epochs=args.epochs, patience=args.epochs, imgsz=args.imgsz, batch=2, device=0,
        optimizer="AdamW", lr0=0.0001, warmup_epochs=3.0, warmup_bias_lr=0.0,
        seed=0, deterministic=True, amp=True, workers=4, cache=False,
        mosaic=0.0, close_mosaic=0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, augmentations=[], bgr=0.0,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.0, multi_scale=0.0,
        project=str(args.output), name=arm, exist_ok=False, plots=True,
    )


def preflight(args):
    """Check native input loading, label alignment, initialization and gradients."""
    import cv2
    from ultralytics.cfg import get_cfg
    from torch.utils._pytree import tree_flatten

    torch.set_num_threads(4)
    torch.manual_seed(0)
    config = yaml.safe_load((DATA / "context.yaml").read_text())
    config["nc"] = 1
    images = sorted((DATA / "images/train").glob("*.tiff"))
    if not images:
        images = sorted((DATA / "images/train").glob("*.tif"))
    assert images, "No context images were prepared"
    for path in images[::max(len(images) // 12, 1)]:
        context = imread(str(path))
        center = cv2.cvtColor(cv2.imread(str(DATA / "single/images/train" / (path.stem + ".png"))), cv2.COLOR_BGR2RGB)
        assert context.ndim == 3 and context.shape[2] == 9
        assert np.array_equal(context[:, :, 3:6], center), str(path)
        original = Path("/home/gonga2/data/racketvision50_yolo/labels/train") / (path.stem + ".txt")
        assert original.read_bytes() == (DATA / "labels/train" / (path.stem + ".txt")).read_bytes()

    source = YOLO(str(SOURCE)).model.float().eval().cuda()
    temporal = DetectionModel(copy.deepcopy(source.yaml), ch=9, nc=1, verbose=False)
    temporal.load(source, verbose=False)
    with torch.no_grad():
        temporal.model[0].conv.weight.zero_()
        temporal.model[0].conv.weight[:, 3:6].copy_(source.model[0].conv.weight)
    temporal = temporal.float().eval().cuda()
    # All layers except the expanded first convolution must start identically.
    for key, value in source.state_dict().items():
        if key != "model.0.conv.weight":
            assert torch.equal(value, temporal.state_dict()[key]), key
    image = imread(str(images[0]))
    image = cv2.resize(image, (args.imgsz, args.imgsz))
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).cuda().float() / 255
    with torch.no_grad():
        a, a_structure = tree_flatten(source(tensor[:, 3:6]))
        b, b_structure = tree_flatten(temporal(tensor))
    assert a_structure == b_structure and len(a) == len(b)
    differences = []
    for left, right in zip(a, b):
        if isinstance(left, torch.Tensor):
            assert left.shape == right.shape
            torch.testing.assert_close(left, right, rtol=1e-4, atol=2e-4)
            differences.append(float((left - right).abs().max()))
    del a, b, source, tensor
    torch.cuda.empty_cache()

    dataset = YOLODataset(img_path=str(DATA / "images/train"), imgsz=args.imgsz,
                          batch_size=2, augment=False, data=config, hyp=get_cfg())
    positive = next(i for i, label in enumerate(dataset.labels) if len(label["cls"]))
    batch = dataset.collate_fn([dataset[positive], dataset[(positive + 1) % len(dataset)]])
    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    batch["img"] = batch["img"].float() / 255
    temporal.args = get_cfg()
    temporal.train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        loss, _ = temporal(batch)
        loss = loss.sum()
    assert torch.isfinite(loss), "Nonfinite preflight loss"
    loss.backward()
    gradient = temporal.model[0].conv.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    norms = [float(gradient[:, start:start + 3].abs().sum()) for start in (0, 3, 6)]
    assert all(value > 0 for value in norms), "Context channels did not receive gradients"
    report = dict(initial_output_max_difference=max(differences), finite_loss=float(loss.detach()),
                  first_layer_gradient_sums=norms, peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                  source_checkpoint=str(SOURCE), source_sha256=digest(SOURCE),
                  channel_order=["previous_RGB", "target_RGB", "next_RGB"], input_shape=list(batch["img"].shape),
                  ultralytics=ultralytics.__version__, torch=torch.__version__)
    save_json(args.output / "preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)


def evaluate(args):
    rows = []
    for arm in ("context", "single"):
        weights = args.output / arm / "weights/best.pt"
        validator = ContextValidator(args=dict(
            data=str(DATA / ("context.yaml" if arm == "context" else "single.yaml")),
            imgsz=args.imgsz, batch=2, device=0, half=False, quantize=None,
            conf=0.001, iou=0.7, max_det=300, workers=4, split="val",
            project=str(args.output / "evaluation"), name=arm, exist_ok=False, plots=True,
        ))
        metrics = validator(model=str(weights))
        csv_rows = list(csv.DictReader((args.output / arm / "results.csv").open()))
        saved_args = yaml.safe_load((args.output / arm / "args.yaml").read_text())
        row = dict(model=arm, weights=str(weights), epochs=len(csv_rows),
                   batch=saved_args["batch"], imgsz=saved_args["imgsz"],
                   precision=float(metrics["metrics/precision(B)"]), recall=float(metrics["metrics/recall(B)"]),
                   mAP50=float(metrics["metrics/mAP50(B)"]), mAP50_95=float(metrics["metrics/mAP50-95(B)"]),
                   inference_ms=float(validator.speed["inference"]))
        rows.append(row)
        save_json(args.output / "comparison.json", rows)
        del validator
        torch.cuda.empty_cache()
    with (args.output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = "| Model | Precision | Recall | mAP50 | mAP50-95 | Inference ms/image |\n|---|---:|---:|---:|---:|---:|\n"
    for row in rows:
        text += f"| {row['model']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['mAP50']:.4f} | {row['mAP50_95']:.4f} | {row['inference_ms']:.2f} |\n"
    text += "\nBoth models continue from the same 50-video single-frame checkpoint. Validation uses 200 frames (193 ball instances), FP32, and the same resolution. Test is not evaluated. Context uses one future frame. No dedicated motion-blur or stationary-distractor subset is labeled.\n"
    (args.output / "comparison.md").write_text(text)
    print(text, flush=True)


def run(args):
    if not (DATA / "COMPLETE.json").is_file():
        raise RuntimeError("Wait for context dataset preparation to finish before training")
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "experiment.json"
    if state_path.exists():
        raise RuntimeError(f"Experiment already exists: {state_path}; refusing to restart or overwrite")
    state = dict(state="preflight", pid=os.getpid(), started_at=time.time(),
                 source_checkpoint=str(SOURCE), source_sha256=digest(SOURCE),
                 script_sha256=digest(Path(__file__)),
                 dataset=str(DATA), offsets=[-1, 0, 1], epochs_per_arm=args.epochs,
                 imgsz=args.imgsz, arms={}, test_evaluated=False)
    save_json(state_path, state)
    try:
        for mode, arm in (("check", None), ("train", "context"), ("train", "single"), ("evaluate", None)):
            state["state"] = arm or mode
            state["updated_at"] = time.time()
            save_json(state_path, state)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", mode,
                       "--output", str(args.output), "--epochs", str(args.epochs), "--imgsz", str(args.imgsz)]
            if arm:
                command += ["--arm", arm]
            started = time.time()
            process = subprocess.Popen(command)
            state["child_pid"] = process.pid
            save_json(state_path, state)
            result = process.wait()
            state["child_pid"] = None
            if result:
                raise RuntimeError(f"{arm or mode} exited with code {result}")
            if arm:
                state["arms"][arm] = dict(elapsed_seconds=time.time() - started, completed_at=time.time())
            save_json(state_path, state)
        state.update(state="complete", finished_at=time.time())
    except BaseException as error:
        state.update(state="failed", error=repr(error), finished_at=time.time())
        raise
    finally:
        save_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["run", "check", "train", "evaluate"], default="run")
    parser.add_argument("--arm", choices=["context", "single"], default="context")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/train/ball50_context3_20261030")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "run":
        run(args)
    elif args.mode == "check":
        preflight(args)
    elif args.mode == "train":
        ContextTrainer(overrides=configuration(args, args.arm)).train()
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
