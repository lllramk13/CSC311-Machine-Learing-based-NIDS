from scipy.stats import randint

from model import load_data, train_model, evaluate_model
from sklearn.tree import DecisionTreeClassifier

def build_model(
        criterion="entropy",
        max_depth=5,
        random_state=42,
):

    return DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
    )

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(sample_percentage=50, target="attack", datasets=["cse_cic_ids2018"])

    model = build_model()

    hyperparams = {
       "max_depth": randint(1, 31),
    }

    results = train_model(
        model=model,
        name="decision_tree_multiclass",
        X_train=X_train,
        y_train=y_train,
        hyperparams=hyperparams,
        n_iter=30,
        scoring="accuracy",
    )

    metrics = evaluate_model(results["model"], X_test, y_test)

    print("Best params:", results["best_params"])
    print("cross validation acc:", results["cv_accuracy"])
    print("test acc:", metrics["accuracy"])
    print("Confusion matrix:", metrics["confusion_matrix"])