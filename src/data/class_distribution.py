import csv
from collections import Counter
from pathlib import Path

from src.config import PROJECT_ROOT, load_config


REPORT_DIR = PROJECT_ROOT / "reports"


def count_classes(csv_path: Path) -> tuple[Counter[str], Counter[str]]:
    """Count labels and attack types in one CSV file."""
    label_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        label_index = header.index("Label")
        attack_index = header.index("Attack")

        for row in reader:
            label_counts[row[label_index]] += 1
            attack_counts[row[attack_index] or "<missing>"] += 1

    return label_counts, attack_counts


def write_distribution(
    output_path: Path,
    rows: list[tuple[str, str, int, float]],
    category_name: str,
) -> None:
    """Write one distribution report."""
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", category_name, "Count", "Percentage"])
        writer.writerows(rows)


def main() -> None:
    """Scan all configured datasets and write distribution reports."""
    config = load_config()
    label_rows: list[tuple[str, str, int, float]] = []
    attack_rows: list[tuple[str, str, int, float]] = []

    for dataset in config["datasets"].values():
        name = dataset["source_name"]
        csv_path = dataset["data_path"]
        print(f"Scanning {name}: {csv_path.name}", flush=True)

        label_counts, attack_counts = count_classes(csv_path)
        total = sum(label_counts.values())

        for label, count in sorted(label_counts.items()):
            label_rows.append((name, label, count, count / total * 100))

        for attack, count in sorted(attack_counts.items()):
            attack_rows.append((name, attack, count, count / total * 100))

        print(f"  rows={total:,}, labels={dict(label_counts)}", flush=True)

    REPORT_DIR.mkdir(exist_ok=True)
    write_distribution(
        REPORT_DIR / "class_distribution.csv",
        label_rows,
        "Label",
    )
    write_distribution(
        REPORT_DIR / "attack_distribution.csv",
        attack_rows,
        "Attack",
    )
    print(f"Reports written to {REPORT_DIR}", flush=True)


if __name__ == "__main__":
    main()
