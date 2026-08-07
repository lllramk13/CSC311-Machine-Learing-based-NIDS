import argparse
from pathlib import Path
import duckdb
from src.config import load_config


def sql_string(value: str | Path):
    return str(value).replace("'", "''")


def convert_dataset(dataset: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.parquet")
    temporary_path.unlink(missing_ok=True)

    csv_path = sql_string(dataset["data_path"])
    parquet_path = sql_string(temporary_path)
    source_name = sql_string(dataset["source_name"])
    temp_directory = sql_string(output_path.parent / ".duckdb_tmp")

    connection = duckdb.connect()
    # connection.execute("SET threads = 1")
    # connection.execute("SET memory_limit = '1GB'")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(f"SET temp_directory = '{temp_directory}'")
    connection.execute(
        f"""
        COPY (
            SELECT *, '{source_name}' AS Dataset
            FROM read_csv('{csv_path}', header = true, auto_detect = true)
        )
        TO '{parquet_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """
    )
    connection.close()
    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset key to convert; may be supplied more than once.",
    )
    args = parser.parse_args()

    config = load_config()
    selected = args.dataset or list(config["datasets"])
    parquet_root = config["processed_root"] / "parquet"

    for key in selected:
        dataset = config["datasets"][key]
        output_path = parquet_root / key / "data.parquet"

        if output_path.exists():
            print(f"Skipping {key}: {output_path.name} already exists")
            continue

        print(f"Converting {key}: {dataset['data_path'].name}", flush=True)
        convert_dataset(dataset, output_path)
        print(
            f"Written {output_path} "
            f"({output_path.stat().st_size / 1_000_000_000:.2f} GB)",
            flush=True,
        )


if __name__ == "__main__":
    main()
