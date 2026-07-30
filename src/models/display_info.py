import matplotlib.pyplot as plt
import seaborn as sns


def display_metrics(metrics):
    """
    Prints evaluation metrics.

    Args:
        metrics (dict): Output from evaluate_model()
    """
    print("Model Performance")
    print("-----------------")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")


def plot_confusion_matrix(metrics, class_names=None):
    """
    Displays confusion matrix as a heatmap.

    Args:
        metrics (dict): Output from evaluate_model()
        class_names (list): Optional class labels
    """
    matrix = metrics["confusion_matrix"]

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()


def display_all(metrics, class_names=None):
    """
    Displays all evaluation information.

    Args:
        metrics (dict): Output from evaluate_model()
        class_names (list): Optional class labels
    """
    display_metrics(metrics)
    plot_confusion_matrix(metrics, class_names)