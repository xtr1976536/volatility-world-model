"""Kaggle entry point for the formal Dow30 experiment."""
from pathlib import Path
import json
import sys
import zipfile
import yaml

ROOT = Path(__file__).resolve().parent
dataset_pkg = next(Path("/kaggle/input").rglob("empirical_runs/world_model_v1"), None)
if dataset_pkg is None:
    embedded = ROOT / "empirical_runs" / "world_model_v1"
    if (embedded / "data" / "dow30").exists():
        dataset_pkg = embedded
if dataset_pkg is None:
    packed = next(Path("/kaggle/input").rglob("empirical_runs.zip"), None)
    if packed is not None:
        unpack_root = Path("/kaggle/working/world_model_source")
        unpack_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(packed) as archive:
            archive.extractall(unpack_root)
        dataset_pkg = unpack_root / "empirical_runs" / "world_model_v1"
if dataset_pkg is None:
    raise FileNotFoundError("Kaggle dataset with empirical_runs/world_model_v1 was not attached")
sys.path.insert(0, str(dataset_pkg.parent.parent))
config_path = dataset_pkg / "config.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["data_source"] = str(dataset_pkg / "data" / "dow30")
config["output_dir"] = "/kaggle/working/dow30_formal"
runtime_config = ROOT / "config_kaggle_runtime.yaml"
runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

from empirical_runs.world_model_v1.evaluate import run

result = run(runtime_config)
print(json.dumps({"status": "complete", "output": str(result)}))
