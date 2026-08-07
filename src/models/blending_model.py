import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from display_info import display_all
from src.data.loader import (
    apply_median_imputer,
    assign_splits,
    fit_median_imputer,
    load_flows,
    model_features,
    replace_non_finite,
    sql_identifier,
)


MODEL_DIR = ROOT / "models"


def load_holdout(sample_percentage):
    flows = load_flows(datasets=["cse_cic_ids2018"])
    flows = assign_splits(replace_non_finite(flows), 80, 0)
    train_flows = flows.filter("Split = 'train'")
    test_flows = flows.filter("Split = 'test'")
    medians = fit_median_imputer(train_flows)
    test_flows = apply_median_imputer(test_flows, medians)
    threshold = int(sample_percentage * 100)

    if threshold < 10_000:
        test_flows = test_flows.filter(
            f"""
            hash(
                Dataset,
                IPV4_SRC_ADDR,
                IPV4_DST_ADDR,
                L4_SRC_PORT,
                L4_DST_PORT,
                PROTOCOL,
                311,
                'sample'
            ) % 10000 < {threshold}
            """
        )

    features = model_features()
    columns = ", ".join(
        sql_identifier(column) for column in [*features, "Attack"]
    )
    frame = test_flows.project(columns).df()
    return frame[features], frame["Attack"]


def make_features(models, X, classes, encoder):
    features = np.empty((len(X), len(models) * len(classes)), dtype=np.float32)

    for index, model in enumerate(models):
        model_classes = np.asarray(model.classes_)
        if np.issubdtype(model_classes.dtype, np.integer):
            model_classes = encoder.inverse_transform(model_classes)

        positions = {name: position for position, name in enumerate(model_classes)}
        order = [positions[name] for name in classes]
        start = index * len(classes)
        end = start + len(classes)
        features[:, start:end] = model.predict_proba(X)[:, order]

    return features


def sample_weights(y, power):
    counts = np.bincount(y)
    return (len(y) / (len(np.unique(y)) * counts[y])) ** power


def print_comparison(names, predictions, y_test, classes):
    print("\nModel comparison")
    for name, prediction in zip(names, predictions):
        accuracy = accuracy_score(y_test, prediction)
        macro_f1 = f1_score(y_test, prediction, average="macro", zero_division=0)
        print(f"{name}: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}")

    print("\nPer-class recall")
    print("Class\t" + "\t".join(names))
    recalls = [
        recall_score(
            y_test,
            prediction,
            labels=np.arange(len(classes)),
            average=None,
            zero_division=0,
        )
        for prediction in predictions
    ]
    for index, class_name in enumerate(classes):
        values = "\t".join(f"{recall[index]:.4f}" for recall in recalls)
        print(f"{class_name}\t{values}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-percentage", type=float, default=20)
    parser.add_argument("--blend-size", type=int, default=400_000)
    parser.add_argument("--validation-size", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    encoder = joblib.load(MODEL_DIR / "xgboost_label_encoder.joblib")
    classes = encoder.classes_
    model_names = ["Decision Tree", "Naive Bayes", "Neural Network", "XGBoost"]
    models = [
        joblib.load(MODEL_DIR / "decision_tree_multiclass.joblib"),
        joblib.load(MODEL_DIR / "naive_bayes.joblib"),
        joblib.load(MODEL_DIR / "neural_network.joblib"),
        joblib.load(MODEL_DIR / "xgboost.joblib"),
    ]

    X_holdout, y_holdout = load_holdout(args.sample_percentage)

    blend_indices, test_indices = train_test_split(
        np.arange(len(y_holdout)),
        train_size=args.blend_size,
        random_state=42,
        stratify=y_holdout,
    )

    X_blend = X_holdout.iloc[blend_indices]
    y_blend = encoder.transform(y_holdout.iloc[blend_indices])
    blend_features = make_features(models, X_blend, classes, encoder)
    del X_blend

    train_indices, validation_indices = train_test_split(
        np.arange(len(y_blend)),
        test_size=args.validation_size,
        random_state=43,
    )

    powers = [0, 0.25, 0.5, 0.75, 1]
    scores = []
    for power in powers:
        candidate = LogisticRegression(max_iter=500)
        candidate.fit(
            blend_features[train_indices],
            y_blend[train_indices],
            sample_weight=sample_weights(y_blend[train_indices], power),
        )
        prediction = candidate.predict(blend_features[validation_indices])
        score = f1_score(
            y_blend[validation_indices],
            prediction,
            average="macro",
            zero_division=0,
        )
        scores.append(score)
        print(f"weight_power={power}: validation_macro_f1={score:.4f}")

    best_power = powers[np.argmax(scores)]
    print("Selected weight power:", best_power)

    blender = LogisticRegression(max_iter=500)
    blender.fit(
        blend_features,
        y_blend,
        sample_weight=sample_weights(y_blend, best_power),
    )
    blender.weight_power_ = best_power
    joblib.dump(blender, MODEL_DIR / "blending_tuned.joblib")
    del y_blend, blend_features

    y_test = np.empty(len(test_indices), dtype=np.int16)
    y_pred = np.empty(len(test_indices), dtype=np.int16)
    base_predictions = np.empty(
        (len(models), len(test_indices)),
        dtype=np.int16,
    )

    for start in range(0, len(test_indices), args.chunk_size):
        end = min(start + args.chunk_size, len(test_indices))
        indices = test_indices[start:end]
        X_chunk = X_holdout.iloc[indices]
        chunk_features = make_features(models, X_chunk, classes, encoder)
        y_test[start:end] = encoder.transform(y_holdout.iloc[indices])
        y_pred[start:end] = blender.predict(chunk_features)
        for model_index in range(len(models)):
            block_start = model_index * len(classes)
            block_end = block_start + len(classes)
            base_predictions[model_index, start:end] = np.argmax(
                chunk_features[:, block_start:block_end],
                axis=1,
            )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=np.arange(len(classes))
        ),
    }

    print("Macro F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))
    print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
    print_comparison(
        [*model_names, "Blending"],
        [*base_predictions, y_pred],
        y_test,
        classes,
    )
    display_all(metrics, class_names=classes)


if __name__ == "__main__":
    main()
