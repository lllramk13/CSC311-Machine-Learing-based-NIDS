import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def load_data():
    data = None 
    return data 

def train_model(
    model,
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
):
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)

    results = {
        "model": model,
        "train_accuracy": train_accuracy,
        "loss_curve": getattr(model, "loss_curve_", None),
    }

    if X_valid is not None and y_valid is not None:
        results["validation_accuracy"] = model.score(X_valid, y_valid)

    return results

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

def predict_model(model, X):
    return model.predict(X)

def plot_history(model):
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, label="Training Loss")

    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.show()