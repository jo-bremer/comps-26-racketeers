"""Shared paths for the YOLO26 scripts. No model or training runs on import."""
import argparse
import importlib.metadata
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_suffix(".json")


def load_config(path=None):
    """Return settings and the folder relative paths are measured from."""
    config_path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
    settings = json.loads(config_path.read_text())
    if not isinstance(settings, dict):
        raise ValueError("The project configuration must be a JSON object.")
    for section in ("data", "model"):
        if section not in settings or not isinstance(settings[section], dict):
            raise ValueError(f"Missing configuration section: {section}")
    return settings, config_path.parent


def resolve_path(base_directory, value):
    """Absolute paths stay absolute; relative paths start at the config folder."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base_directory) / path).resolve()


def dataset_root(settings, base_directory, dataset="full"):
    if dataset not in ("full", "pilot"):
        raise ValueError("Dataset must be full or pilot.")
    key = "prepared_root" if dataset == "full" else "pilot_root"
    return resolve_path(base_directory, settings["data"][key])


def write_dataset_yaml(settings, base_directory, dataset="full"):
    """Make the library's dataset YAML agree with the one editable config."""
    root = dataset_root(settings, base_directory, dataset)
    filename = "ball.yaml" if dataset == "full" else "ball50.yaml"
    path = Path(base_directory) / filename
    content = (
        f"path: {json.dumps(str(root))}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "\nnames:\n  0: ball\n"
    )
    if not path.exists() or path.read_text() != content:
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(content)
        temporary.replace(path)
    return path


def check_inputs(settings, base_directory):
    """Inspect filesystem/package metadata only; do not import ML libraries."""
    raw = resolve_path(base_directory, settings["data"]["raw_root"])
    pretrained = resolve_path(base_directory, settings["model"]["pretrained"])
    required = [raw / "videos", raw / "all", pretrained]
    required.extend(raw / "info" / f"{split}.json" for split in ("train", "val", "test"))
    report = {
        "config_directory": str(base_directory),
        "required_inputs": {str(path): path.exists() for path in required},
        "prepared_datasets": {
            name: str(dataset_root(settings, base_directory, name))
            for name in ("full", "pilot")
        },
        "saved_checkpoints": {
            name: {"path": str(resolve_path(base_directory, value)),
                   "exists": resolve_path(base_directory, value).is_file()}
            for name, value in settings.get("checkpoints", {}).items()
        },
        "packages": {},
    }
    for name in ("ultralytics", "torch", "torchvision", "opencv-python", "numpy", "PyYAML", "Pillow"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    report["required_inputs_present"] = all(report["required_inputs"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Default: project_config.json beside this file")
    parser.add_argument("--check", action="store_true", help="Inspect paths and package versions without loading models")
    args = parser.parse_args()
    if not args.check:
        parser.print_help()
        return
    settings, base = load_config(args.config)
    report = check_inputs(settings, base)
    print(json.dumps(report, indent=2))
    if not report["required_inputs_present"]:
        raise SystemExit("Required inputs are missing. Edit project_config.json before running the workflow.")


if __name__ == "__main__":
    main()
