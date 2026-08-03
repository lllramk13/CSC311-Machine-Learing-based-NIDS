from pathlib import Path

import joblib
from scipy.stats import randint, loguniform

from model import load_data, train_model, evaluate_model
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def build_model(
    max_depth = 5,
    random_state = 42,
):
    return XGBClassifier(
        max_depth=max_depth,
        random_state=random_state,
        tree_method="hist",
        device="cuda",
    )

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(
        datasets=["cse_cic_ids2018"],
        target="attack",
        sample_percentage=50,)

    # XGBoost multiclass targets must be consecutive integers starting at zero.
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

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
            scoring="roc_auc_ovr_weighted",
        )

    model_dir = Path(__file__).resolve().parents[2] / "models"
    joblib.dump(label_encoder, model_dir / "xgboost_label_encoder.joblib")

    metrics = evaluate_model(results["model"], X_test, y_test)

    print("Best params:", results["best_params"])
    print("CV weighted ROC AUC:", results["cv_accuracy"])
    print("test acc:", metrics["accuracy"])
    print("Class order:", label_encoder.classes_)
    print("Confusion matrix:", metrics["confusion_matrix"])
