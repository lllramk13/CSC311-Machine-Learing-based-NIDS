from scipy.stats import randint, loguniform

from model import load_data, train_model, evaluate_model
from xgboost import XGBClassifier

def build_model(
    max_depth = 5,
    random_state = 42,
):
    return XGBClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    model = build_model()

    hyperparams = {
            "max_depth": randint(3, 8),
            "n_estimators": randint(50, 250),
            "learning_rate": loguniform(0.01, 0.1),
        }

    results = train_model(
            model=model,
            name="xgboost",
            X_train=X_train,
            y_train=y_train,
            hyperparams=hyperparams,
            n_iter=50,
            scoring="roc_auc",
        )

    metrics = evaluate_model(results["model"], X_test, y_test)

    print("Best params:", results["best_params"])
    print("cross validation acc:", results["cv_accuracy"])
    print("test acc:", metrics["accuracy"])
    print("Confusion matrix:", metrics["confusion_matrix"])