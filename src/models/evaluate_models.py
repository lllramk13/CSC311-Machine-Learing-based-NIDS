import argparse
import gc
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.lib.format import open_memmap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.blending_model import load_holdout
from src.models.display_info import plot_confusion_matrix


MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATHS = {
    "Decision Tree": MODEL_DIR / "decision_tree_multiclass.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Neural Network": MODEL_DIR / "neural_network.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
}
PLOT_NAMES = {
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
    "neural_network": "Neural Network",
    "xgboost": "XGBoost",
    "blending": "Blending",
    "weighted_voting": "Weighted Soft Voting",
}
SHORT_CLASSES = [
    "Benign",
    "Bot",
    "Brute-Web",
    "Brute-XSS",
    "DDoS-HOIC",
    "DDoS-LOIC-UDP",
    "DDoS-LOIC-HTTP",
    "DoS-GoldenEye",
    "DoS-Hulk",
    "DoS-SlowHTTP",
    "DoS-Slowloris",
    "FTP-BruteForce",
    "Infiltration",
    "SQL-Injection",
    "SSH-Bruteforce",
]


def aligned_proba(model, X, classes, encoder):
    model_classes = np.asarray(model.classes_)
    if np.issubdtype(model_classes.dtype, np.integer):
        model_classes = encoder.inverse_transform(model_classes)
    positions = {name: index for index, name in enumerate(model_classes)}
    order = [positions[name] for name in classes]
    return model.predict_proba(X)[:, order]


def evaluate(y_true, y_pred, classes):
    observed = np.unique(y_true)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=observed,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": recall_score(
            y_true,
            y_pred,
            labels=observed,
            average="macro",
            zero_division=0,
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "support": np.bincount(y_true, minlength=len(classes)),
        "recall_per_class": recall_score(
            y_true,
            y_pred,
            labels=np.arange(len(classes)),
            average=None,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(len(classes))
        ),
    }


def weighted_prediction(paths, indices, weights, class_count):
    scores = np.zeros((len(indices), class_count), dtype=np.float32)
    for path, weight in zip(paths, weights):
        scores += weight * np.load(path, mmap_mode="r")[indices]
    return scores.argmax(axis=1)


def blender_prediction(blender, paths, indices, chunk_size):
    prediction = np.empty(len(indices), dtype=np.int16)
    probabilities = [np.load(path, mmap_mode="r") for path in paths]
    for start in range(0, len(indices), chunk_size):
        end = min(start + chunk_size, len(indices))
        rows = indices[start:end]
        features = np.hstack([values[rows] for values in probabilities])
        prediction[start:end] = blender.predict(features)
    return prediction


def format_report(results, weights, ablations, classes):
    names = list(results)
    lines = [
        "Unified Model Evaluation",
        "========================",
        "",
        "Weighted Soft Voting weights",
    ]
    lines.extend(
        f"{name}: {weight:.6f}"
        for name, weight in zip(MODEL_PATHS, weights)
    )
    lines.extend(["", "Overall metrics"])
    lines.append(
        "Model\tAccuracy\tMacro F1\tBalanced Accuracy\t"
        "Weighted Precision\tWeighted Recall\tWeighted F1"
    )
    for name, values in results.items():
        lines.append(
            f"{name}\t{values['accuracy']:.6f}\t{values['macro_f1']:.6f}\t"
            f"{values['balanced_accuracy']:.6f}\t"
            f"{values['precision_weighted']:.6f}\t"
            f"{values['recall_weighted']:.6f}\t"
            f"{values['f1_weighted']:.6f}"
        )
    lines.extend(["", "Leave-one-model-out weighted voting"])
    lines.append("Omitted model\tAccuracy\tMacro F1\tBalanced Accuracy")
    for name, values in ablations.items():
        lines.append(
            f"{name}\t{values['accuracy']:.6f}\t{values['macro_f1']:.6f}\t"
            f"{values['balanced_accuracy']:.6f}"
        )
    lines.extend(["", "Per-class recall"])
    lines.append("Class\tSupport\t" + "\t".join(names))
    support = results[names[0]]["support"]
    for index, class_name in enumerate(classes):
        recalls = "\t".join(
            f"{results[name]['recall_per_class'][index]:.6f}"
            for name in names
        )
        lines.append(f"{class_name}\t{support[index]}\t{recalls}")
    for name, values in results.items():
        lines.extend(
            [
                "",
                f"{name} confusion matrix",
                np.array2string(values["confusion_matrix"]),
            ]
        )
    return "\n".join(lines)


def run_evaluation(sample_percentage, blend_size, chunk_size):
    encoder = joblib.load(MODEL_DIR / "xgboost_label_encoder.joblib")
    classes = encoder.classes_
    X, y = load_holdout(sample_percentage)
    blend_indices, test_indices = train_test_split(
        np.arange(len(y)),
        train_size=blend_size,
        random_state=42,
        stratify=y,
    )
    y_encoded = encoder.transform(y)
    y_blend = y_encoded[blend_indices]
    y_test = y_encoded[test_indices]
    results = {}
    validation_scores = []

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        probability_paths = []
        for model_index, (name, model_path) in enumerate(MODEL_PATHS.items()):
            print(f"Predicting with {name}...", flush=True)
            model = joblib.load(model_path)
            path = Path(directory) / f"model_{model_index}.npy"
            probabilities = open_memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=(len(X), len(classes)),
            )
            for start in range(0, len(X), chunk_size):
                end = min(start + chunk_size, len(X))
                probabilities[start:end] = aligned_proba(
                    model, X.iloc[start:end], classes, encoder
                )
            probabilities.flush()
            validation_prediction = probabilities[blend_indices].argmax(axis=1)
            validation_scores.append(
                f1_score(
                    y_blend,
                    validation_prediction,
                    labels=np.unique(y_blend),
                    average="macro",
                    zero_division=0,
                )
            )
            test_prediction = probabilities[test_indices].argmax(axis=1)
            results[name] = evaluate(y_test, test_prediction, classes)
            probability_paths.append(path)
            del model, probabilities
            gc.collect()

        weights = np.asarray(validation_scores)
        weights /= weights.sum()
        blender = joblib.load(MODEL_DIR / "blending.joblib")
        blend_prediction = blender_prediction(
            blender, probability_paths, test_indices, chunk_size
        )
        results["Blending"] = evaluate(y_test, blend_prediction, classes)
        voting_prediction = weighted_prediction(
            probability_paths, test_indices, weights, len(classes)
        )
        results["Weighted Soft Voting"] = evaluate(
            y_test, voting_prediction, classes
        )

        ablations = {}
        for omitted, name in enumerate(MODEL_PATHS):
            ablation_weights = weights.copy()
            ablation_weights[omitted] = 0
            prediction = weighted_prediction(
                probability_paths,
                test_indices,
                ablation_weights,
                len(classes),
            )
            ablations[name] = evaluate(y_test, prediction, classes)

    REPORT_DIR.mkdir(exist_ok=True)
    rows = [
        {
            "model": name,
            **{
                key: values[key]
                for key in (
                    "accuracy",
                    "macro_f1",
                    "balanced_accuracy",
                    "precision_weighted",
                    "recall_weighted",
                    "f1_weighted",
                )
            },
        }
        for name, values in results.items()
    ]
    pd.DataFrame(rows).to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    report = format_report(results, weights, ablations, classes)
    (REPORT_DIR / "model_evaluation.txt").write_text(report, encoding="utf-8")
    joblib.dump(
        {
            "classes": classes,
            "results": results,
            "weights": weights,
            "ablations": ablations,
        },
        REPORT_DIR / "model_evaluation.joblib",
    )
    joblib.dump(
        {"model_names": list(MODEL_PATHS), "weights": weights},
        MODEL_DIR / "weighted_voting.joblib",
    )
    print(report)


def show_plot(name):
    saved = joblib.load(REPORT_DIR / "model_evaluation.joblib")
    if name == "all":
        fig, axes = plt.subplots(1, 6, figsize=(32, 7))
        for index, (ax, model_name) in enumerate(
            zip(axes, PLOT_NAMES.values())
        ):
            matrix = saved["results"][model_name]["confusion_matrix"]
            totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                totals,
                out=np.zeros_like(matrix, dtype=float),
                where=totals != 0,
            )
            sns.heatmap(
                normalized,
                ax=ax,
                cmap="Blues",
                vmin=0,
                vmax=1,
                cbar=False,
                xticklabels=SHORT_CLASSES,
                yticklabels=SHORT_CLASSES,
            )
            ax.set_title(model_name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual" if index == 0 else "")
            ax.tick_params(axis="x", rotation=60, labelsize=6)
            ax.tick_params(axis="y", rotation=0, labelsize=6)
            if index:
                ax.set_yticklabels([])
        colorbar_axis = fig.add_axes([0.93, 0.28, 0.008, 0.5])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="Blues"),
            cax=colorbar_axis,
        )
        colorbar.set_label("Row-normalized proportion")
        fig.suptitle("Normalized Confusion Matrices", fontsize=18)
        fig.subplots_adjust(left=0.07, right=0.92, bottom=0.27, top=0.86, wspace=0.08)
        plt.savefig(
            REPORT_DIR / "all_confusion_matrices.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
    else:
        model_name = PLOT_NAMES[name]
        print(model_name)
        plot_confusion_matrix(
            {"confusion_matrix": saved["results"][model_name]["confusion_matrix"]},
            saved["classes"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-percentage", type=float, default=20)
    parser.add_argument("--blend-size", type=int, default=400_000)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--plot", choices=[*PLOT_NAMES, "all"])
    args = parser.parse_args()
    if args.plot:
        show_plot(args.plot)
    else:
        run_evaluation(
            args.sample_percentage,
            args.blend_size,
            args.chunk_size,
        )


if __name__ == "__main__":
    main()
