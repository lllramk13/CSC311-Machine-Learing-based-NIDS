import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib

def load_data():
    data = None 
    return data 

def train_model(
    model,
    name,
    X_train,
    y_train,
    hyperparams,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
):
    # name: what to save model as 
    # TODO: Change n_jobs to desired amount
    # TODO: Put code to download model.
    # cv: cross validation number 
    # scoring: metric to see what is the best model

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=hyperparams,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=1,
        refit=True,
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    results = {
        "model": best_model,
        "train_accuracy": best_model.score(X_train, y_train),
        "cv_accuracy": random_search.best_score_,
        "best_params": random_search.best_params_,
        "loss_curve": getattr(best_model, "loss_curve_", None),
    }

    joblib.dump(best_model, f"models/{name}.joblib")

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