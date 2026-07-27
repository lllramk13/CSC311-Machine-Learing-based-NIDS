import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "dataset.local.json"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """load configuration and resolve dataset paths."""
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    data_root = Path(config["data_root"]).expanduser()

    for dataset in config["datasets"].values():
        dataset_root = data_root / dataset["root"]
        dataset["data_path"] = (
            dataset_root / dataset["data_path"].lstrip("/\\")
        ).resolve()
        dataset["feature_path"] = (
            dataset_root / dataset["feature_path"].lstrip("/\\")
        ).resolve()

    config["processed_root"] = Path(
        config["processed_root"]
    ).expanduser().resolve()

    return config


def load_features(config: dict | None = None) -> list[str]:
    """Load the shared NetFlow V3 feature names."""
    config = config or load_config()
    dataset = next(iter(config["datasets"].values()))

    with dataset["feature_path"].open(encoding="utf-8-sig", newline="") as file:
        return [row["Feature"].strip() for row in csv.DictReader(file)]
