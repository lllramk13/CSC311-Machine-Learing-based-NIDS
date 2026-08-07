"graph and table generation only"
import gc
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = REPORT_DIR / "report_artifacts"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
MODEL_NAMES = [
    "Decision Tree",
    "Naive Bayes",
    "Neural Network",
    "XGBoost",
]
MODEL_PATHS = [
    MODEL_DIR / "decision_tree_multiclass.joblib",
    MODEL_DIR / "naive_bayes.joblib",
    MODEL_DIR / "neural_network.joblib",
    MODEL_DIR / "xgboost.joblib",
]
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
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def add_box(ax, x, y, width, height, text, color):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11)


def add_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#333333"},
    )


def plot_workflow():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")
    add_box(ax, 5.5, 8.4, 4, 1, "NF-CICIDS2018-v3\n49 flow features", "#E8F1FA")
    add_box(ax, 1, 6.2, 4, 1, "Base-model training pool\n80% deterministic split", "#EAF4E2")
    add_box(ax, 10, 6.2, 4, 1, "Original holdout\n20% deterministic split", "#FFF0D9")
    add_box(ax, 0.3, 3.4, 2.5, 1.4, "Decision Tree\nGaussianNB", "#F1F1F1")
    add_box(ax, 3.2, 3.4, 2.5, 1.4, "Neural Network\nXGBoost", "#F1F1F1")
    add_box(ax, 8.1, 3.4, 3, 1.4, "Ensemble design set\n400,000 flows", "#FBE3C2")
    add_box(ax, 11.8, 3.4, 3, 1.4, "Final test set\n408,197 flows", "#F8D7DA")
    add_box(ax, 7, 0.7, 3.2, 1.4, "Logistic-regression\nBlending", "#E7DDF2")
    add_box(ax, 10.7, 0.7, 3.2, 1.4, "Validation-weighted\nSoft Voting", "#DDF0ED")
    add_arrow(ax, (7.5, 8.4), (3, 7.2))
    add_arrow(ax, (7.5, 8.4), (12, 7.2))
    add_arrow(ax, (3, 6.2), (1.55, 4.8))
    add_arrow(ax, (3, 6.2), (4.45, 4.8))
    add_arrow(ax, (12, 6.2), (9.6, 4.8))
    add_arrow(ax, (12, 6.2), (13.3, 4.8))
    add_arrow(ax, (9.6, 3.4), (8.6, 2.1))
    add_arrow(ax, (9.6, 3.4), (12.3, 2.1))
    ax.text(13.3, 2.85, "Common evaluation\nof all six methods", ha="center", va="center", fontsize=10)
    ax.set_title("Experimental Workflow", fontsize=18, pad=15)
    save_figure(OUTPUT_DIR / "figure_1_workflow.png")


def plot_class_distribution(classes, support):
    plt.figure(figsize=(14, 6))
    bars = plt.bar(SHORT_CLASSES, support, color="#4C78A8")
    plt.yscale("symlog", linthresh=1)
    plt.ylabel("Number of final-test flows (symlog scale)")
    plt.xlabel("Traffic class")
    plt.title(f"Final-Test Class Distribution (n = {support.sum():,})")
    plt.xticks(rotation=45, ha="right")
    for bar, value in zip(bars, support):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            max(value, 0.6),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )
    save_figure(OUTPUT_DIR / "figure_2_class_distribution.png")
    pd.DataFrame({"class": classes, "support": support}).to_csv(
        OUTPUT_DIR / "table_class_support.csv", index=False
    )


def plot_metric_comparison(results):
    metrics = ["accuracy", "f1_weighted", "macro_f1", "balanced_accuracy"]
    labels = ["Accuracy", "Weighted F1", "Macro F1", "Balanced Accuracy"]
    names = list(results)
    x = np.arange(len(names))
    width = 0.19
    plt.figure(figsize=(15, 7))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        values = [results[name][metric] for name in names]
        plt.bar(x + (index - 1.5) * width, values, width, label=label)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Performance Comparison on the Common Final Test Set", pad=55)
    plt.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    plt.grid(axis="y", alpha=0.25)
    save_figure(OUTPUT_DIR / "figure_3_model_comparison.png")


def normalized_matrix(matrix):
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)


def matrix_labels(matrix):
    return np.where(matrix >= 0.005, np.vectorize(lambda value: f"{value:.2f}")(matrix), "")


def draw_matrix(ax, matrix, title, labels, annotations=True, colorbar=True):
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=matrix_labels(matrix) if annotations else False,
        fmt="",
        annot_kws={"fontsize": 6},
        xticklabels=labels,
        yticklabels=labels,
        cbar=colorbar,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=55, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)


def plot_confusion_matrices(results, classes):
    normalized = {}
    for name, values in results.items():
        matrix = values["confusion_matrix"]
        normalized[name] = normalized_matrix(matrix)
        slug = name.lower().replace(" ", "_")
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
            CONFUSION_DIR / f"{slug}_counts.csv"
        )
        fig, ax = plt.subplots(figsize=(12, 10))
        draw_matrix(ax, normalized[name], f"{name} Normalized Confusion Matrix", SHORT_CLASSES)
        save_figure(CONFUSION_DIR / f"{slug}_normalized.png")

    main_names = ["XGBoost", "Blending", "Weighted Soft Voting"]
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    for ax, name in zip(axes, main_names):
        draw_matrix(ax, normalized[name], name, SHORT_CLASSES, annotations=False)
    save_figure(OUTPUT_DIR / "figure_4_main_confusion_matrices.png")

    fig, axes = plt.subplots(2, 3, figsize=(28, 18))
    for ax, (name, matrix) in zip(axes.flat, normalized.items()):
        draw_matrix(ax, matrix, name, SHORT_CLASSES, annotations=False)
    fig.suptitle("Normalized Confusion Matrices of All Six Methods", fontsize=20)
    save_figure(CONFUSION_DIR / "all_models_normalized.png")


def plot_ablation(saved):
    full = saved["results"]["Weighted Soft Voting"]["macro_f1"]
    names = ["Full ensemble", *[f"Without {name}" for name in MODEL_NAMES]]
    values = [full, *[saved["ablations"][name]["macro_f1"] for name in MODEL_NAMES]]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, values, color=["#4C78A8", "#A0CBE8", "#A0CBE8", "#A0CBE8", "#A0CBE8"])
    plt.axvline(full, color="#E45756", linestyle="--", label=f"Full ensemble = {full:.4f}")
    plt.xlim(min(values) - 0.01, max(values) + 0.01)
    plt.xlabel("Macro F1")
    plt.title("Leave-One-Model-Out Weighted Voting")
    plt.legend()
    for bar, value in zip(bars, values):
        plt.text(value + 0.0005, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    save_figure(OUTPUT_DIR / "figure_5_leave_one_out.png")


def save_split_table():
    pd.DataFrame(
        [
            ["Base-model training pool", "Train and tune base classifiers", "80% deterministic split"],
            ["Original holdout", "Excluded from base-model training", "20% deterministic split"],
            ["Ensemble design set", "Train Blending and determine voting weights", "400,000"],
            ["Final test set", "Evaluate all six methods", "408,197"],
        ],
        columns=["partition", "purpose", "size"],
    ).to_csv(OUTPUT_DIR / "table_1_split_summary.csv", index=False)


def save_hyperparameter_table():
    rows = []
    selected = [
        ["criterion", "max_depth", "min_samples_split", "min_samples_leaf"],
        ["var_smoothing"],
        ["hidden_layer_sizes", "activation", "alpha", "batch_size", "learning_rate_init", "early_stopping"],
        ["learning_rate", "max_depth", "n_estimators", "tree_method", "device"],
    ]
    for name, path, keys in zip(MODEL_NAMES, MODEL_PATHS, selected):
        model = joblib.load(path)
        parameters = model.get_params()
        rows.append([name, "; ".join(f"{key}={parameters[key]}" for key in keys)])
        del model
        gc.collect()
    blender = joblib.load(MODEL_DIR / "blending.joblib")
    rows.append(["Blending", f"LogisticRegression; max_iter={blender.get_params()['max_iter']}"])
    rows.append(["Weighted Soft Voting", "Weights proportional to design-set Macro F1"])
    pd.DataFrame(rows, columns=["model", "selected_configuration"]).to_csv(
        OUTPUT_DIR / "table_2_hyperparameters.csv", index=False
    )


def save_metric_tables(saved):
    results = saved["results"]
    metric_names = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    pd.DataFrame(
        [[name, *[values[key] for key in metric_names]] for name, values in results.items()],
        columns=["model", *metric_names],
    ).to_csv(OUTPUT_DIR / "table_3_overall_metrics.csv", index=False)

    full = results["Weighted Soft Voting"]["macro_f1"]
    weights = saved["weights"]
    rows = []
    for name, weight in zip(MODEL_NAMES, weights):
        ablation = saved["ablations"][name]
        rows.append(
            [
                name,
                weight,
                ablation["accuracy"],
                ablation["macro_f1"],
                ablation["macro_f1"] - full,
                ablation["balanced_accuracy"],
            ]
        )
    pd.DataFrame(
        rows,
        columns=[
            "omitted_model",
            "voting_weight",
            "accuracy_when_omitted",
            "macro_f1_when_omitted",
            "macro_f1_change",
            "balanced_accuracy_when_omitted",
        ],
    ).to_csv(OUTPUT_DIR / "table_4_weights_and_ablation.csv", index=False)


def save_recall_tables(saved):
    classes = saved["classes"]
    results = saved["results"]
    names = list(results)
    support = results[names[0]]["support"]
    rows = []
    for index, class_name in enumerate(classes):
        rows.append(
            [
                class_name,
                support[index],
                *[results[name]["recall_per_class"][index] for name in names],
            ]
        )
    table = pd.DataFrame(rows, columns=["class", "support", *names])
    table.to_csv(OUTPUT_DIR / "table_full_class_recall.csv", index=False)
    difficult = [
        "Brute_Force_-Web",
        "Brute_Force_-XSS",
        "DoS_attacks-SlowHTTPTest",
        "Infilteration",
        "SQL_Injection",
    ]
    table[table["class"].isin(difficult)].to_csv(
        OUTPUT_DIR / "table_5_difficult_class_recall.csv", index=False
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    CONFUSION_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved = joblib.load(REPORT_DIR / "model_evaluation.joblib")
    classes = saved["classes"]
    results = saved["results"]
    support = results["Decision Tree"]["support"]
    plot_workflow()
    plot_class_distribution(classes, support)
    plot_metric_comparison(results)
    plot_confusion_matrices(results, classes)
    plot_ablation(saved)
    save_split_table()
    save_hyperparameter_table()
    save_metric_tables(saved)
    save_recall_tables(saved)
    print(f"Report artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
