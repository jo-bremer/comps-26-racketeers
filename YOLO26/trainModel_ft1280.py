"""Fine-tune YOLO26 at image size 1280 using the project configuration."""

import argparse
from pathlib import Path

from project_config import load_config, resolve_path, dataset_root, write_dataset_yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Project JSON; default is beside this script")
    parser.add_argument("--dataset", choices=("full", "pilot"), default="full")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch limit")
    parser.add_argument("--name", help="New run-directory name; existing directories are protected")
    args = parser.parse_args(argv)
    config, base = load_config(args.config)
    settings = dict(config["training"])
    if args.dataset == "pilot":
        settings.update(settings.get("pilot_overrides", {}))
    epochs = args.epochs if args.epochs is not None else settings["epochs"]
    if epochs <= 0:
        parser.error("epochs must be positive")
    pretrained = resolve_path(base, config["model"]["pretrained"])
    data_root = dataset_root(config, base, args.dataset)
    project = resolve_path(base, settings["project"])
    name = args.name or settings["names"][args.dataset + "1280"]
    if not name or Path(name).name != name or name in (".", ".."):
        parser.error("name must be a single directory name")
    if (project / name).exists():
        parser.error(f"Run already exists: {project / name}; choose a new --name")
    if not pretrained.is_file():
        parser.error(f"Missing pretrained checkpoint: {pretrained}")
    for split in ("train", "val"):
        if not (data_root / "images" / split).is_dir() or not (data_root / "labels" / split).is_dir():
            parser.error(f"Missing prepared {split} data: {data_root}; run runner.py first")
    yaml_path = write_dataset_yaml(config, base, args.dataset)
    options = dict(
        data=str(yaml_path), epochs=epochs, patience=settings["patience"], imgsz=1280,
        batch=settings["batch"], device=config["runtime"]["device"],
        optimizer=settings["optimizer"], lr0=settings["lr0"],
        warmup_epochs=settings.get("warmup_epochs", 3.0),
        warmup_bias_lr=settings["warmup_bias_lr"], seed=settings["seed"],
        project=str(project), name=name, exist_ok=False, resume=False,
    )
    nms = settings.get("nms")
    if nms is not None:
        if type(nms) is not bool:
            parser.error("training.nms must be true, false, or null")
        options["nms"] = nms
    from ultralytics import YOLO

    model = YOLO(str(pretrained))
    model.train(**options)
    print(f"Training output: {model.trainer.save_dir}")
    print(f"Best checkpoint: {model.trainer.best}")
    print(f"Last checkpoint: {model.trainer.last}")


if __name__ == "__main__":
    main()
