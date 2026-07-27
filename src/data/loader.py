from collections.abc import Iterable
import duckdb
from src.config import load_config, load_features


IP_COLUMNS = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR"}
TIMESTAMP_COLUMNS = {
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
}
METADATA_COLUMNS = [
    "Attack",
    "Dataset",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
]


def sql_string(value: str) -> str:
    return value.replace("'", "''")


def sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def model_features(
    include_ip: bool = False,
    include_timestamps: bool = False,
) -> list[str]:
    # can add more if needed

    features = load_features()

    if not include_ip:
        features = [name for name in features if name not in IP_COLUMNS]
    if not include_timestamps:
        features = [
            name for name in features if name not in TIMESTAMP_COLUMNS
        ]

    return features


def load_flows(
    datasets: Iterable[str] | None = None,
    labels: Iterable[int] | None = None,
    attacks: Iterable[str] | None = None,
) -> duckdb.DuckDBPyRelation:

    config = load_config()
    selected = set(datasets or config["datasets"])
    parquet_root = config["processed_root"] / "parquet"
    paths = [
        str(parquet_root / key / "data.parquet")
        for key in config["datasets"]
        if key in selected
    ]

    flows = duckdb.read_parquet(paths)

    if labels is not None:
        values = ", ".join(str(int(label)) for label in labels)
        flows = flows.filter(f"Label IN ({values})")

    if attacks is not None:
        values = ", ".join(
            f"'{sql_string(attack)}'" for attack in attacks
        )
        flows = flows.filter(f"Attack IN ({values})")

    return flows


def training_views(
    flows: duckdb.DuckDBPyRelation,
    include_ip: bool = False,
    include_timestamps: bool = False,
) -> tuple[
    duckdb.DuckDBPyRelation,
    duckdb.DuckDBPyRelation,
    duckdb.DuckDBPyRelation,
]:
    features = model_features(include_ip, include_timestamps)
    feature_sql = ", ".join(sql_identifier(name) for name in features)
    metadata_sql = ", ".join(
        sql_identifier(name) for name in METADATA_COLUMNS
    )

    return (
        flows.project(feature_sql),
        flows.project("Label"),
        flows.project(metadata_sql),
    )


def assign_splits(
    flows,
    train_percentage: int = 70,
    validation_percentage: int = 15,
):
    """Assign flows to deterministic train, validation, and test splits."""
    split_number = """
        hash(
            Dataset,
            IPV4_SRC_ADDR,
            IPV4_DST_ADDR,
            L4_SRC_PORT,
            L4_DST_PORT,
            PROTOCOL,
            311
        ) % 100
    """

    return flows.project(
        f"""
        *,
        CASE
            WHEN {split_number} < {train_percentage}
                THEN 'train'
            WHEN {split_number} <
                 {train_percentage + validation_percentage}
                THEN 'validation'
            ELSE 'test'
        END AS Split
        """
    )
