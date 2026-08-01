"""
- python -m pip install -r requirements.txt
- creat config/dataset.local.json, and just change the root to the path of the dataset folder unziped
and the processed to the processed folder path in the dataset folder. the whole thing is gonna to be about 15gb
- to use:   from src.data.get_data import prepare_sklearn_data
            X_train, X_test, y_train, y_test = prepare_sklearn_data(
                datasets=[""], any combination between "unsw_nb15", "ton_iot", "bot_iot", "cse_cic_ids2018"
                labels=[0, 1], 0 is only benign flow, 1 is only attack flow, [0, 1] is all flow
                attack=[""], if no Benign and only attack type, then the label will only be 1, since they are all attack.
                  choice attack type: Benign, Bot, Brute_Force_-Web, Brute_Force_-XSS, DDOS_attack-HOIC, DDOS_attack-LOIC-UDP, DDoS_attacks-LOIC-HTTP, DoS_attacks-GoldenEye, DoS_attacks-Hulk, DoS_attacks-SlowHTTPTest, DoS_attacks-Slowloris, FTP-BruteForce, Infilteration, SQL_Injection, SSH-Bruteforce
                target="label" for benign/attack binary classification, or
                  target="attack" for attack-type multiclass classification
                train_percentage=int, if 70 then 70% train, 30 =% test
                sample_percentage=int, =100 means use all data of chosen above
                include_ip=bool, dataset all have 54 features, i romved ip and timestamp as they may overfit our model but you can add them back here if you guys needed it
                include_timestamps=bool
            )
- ex: 
    datasets=["cse_cic_ids2018"]
    labels=[0, 1]
    attacks=None
    train_percentage=80
    sample_percentage=5.0
    include_ip=False
    include_timestamps=False
"""
from collections.abc import Iterable
import pandas as pd
from src.data.loader import (
    apply_median_imputer,
    assign_splits,
    fit_median_imputer,
    load_flows,
    model_features,
    replace_non_finite,
    sql_identifier,
)


def prepare_sklearn_data(
    datasets: Iterable[str] | None = None,
    labels: Iterable[int] | None = None,
    attacks: Iterable[str] | None = None,
    target: str = "label",
    train_percentage: int = 80,
    sample_percentage: float = 1.0,
    include_ip: bool = False,
    include_timestamps: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target_columns = {
        "label": "Label",
        "attack": "Attack",
    }
    try:
        target_column = target_columns[target.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError("target must be either 'label' or 'attack'") from error

    flows = load_flows(
        datasets=datasets,
        labels=labels,
        attacks=attacks,
    )
    flows = replace_non_finite(flows)
    flows = assign_splits(
        flows,
        train_percentage=train_percentage,
        validation_percentage=0,
    )

    train_flows = flows.filter("Split = 'train'")
    test_flows = flows.filter("Split = 'test'")

    medians = fit_median_imputer(train_flows)
    train_flows = apply_median_imputer(train_flows, medians)
    test_flows = apply_median_imputer(test_flows, medians)

    sample_threshold = int(sample_percentage * 100)
    if sample_threshold < 10_000:
        sample_filter = f"""
            hash(
                Dataset,
                IPV4_SRC_ADDR,
                IPV4_DST_ADDR,
                L4_SRC_PORT,
                L4_DST_PORT,
                PROTOCOL,
                311,
                'sample'
            ) % 10000 < {sample_threshold}
        """
        train_flows = train_flows.filter(sample_filter)
        test_flows = test_flows.filter(sample_filter)

    features = model_features(
        include_ip=include_ip,
        include_timestamps=include_timestamps,
    )
    selected_columns = ", ".join(
        sql_identifier(column) for column in [*features, target_column]
    )

    train_frame = train_flows.project(selected_columns).df()
    test_frame = test_flows.project(selected_columns).df()

    X_train = train_frame[features]
    X_test = test_frame[features]
    y_train = train_frame[target_column]
    y_test = test_frame[target_column]

    return X_train, X_test, y_train, y_test
