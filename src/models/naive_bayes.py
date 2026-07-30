from scipy.stats import loguniform

from model import load_data, train_model, evaluate_model
from sklearn.naive_bayes import GaussianNB

def build_model(
    var_smoothing=1e-9,
    priors=None,
):
    return GaussianNB(
        var_smoothing=var_smoothing,
        priors=priors,
    )


if __name__ == "__main__":
    print("I am running!")

    X_train, y_train, X_test, y_test = load_data()

    model = build_model()

    hyperparams = {
        "var_smoothing": loguniform(1e-12, 1e-7),
    }

    results = train_model(
        model=model,
        name="naive_bayes",
        X_train=X_train,
        y_train=y_train,
        hyperparams=hyperparams,
        n_iter=10,
    )

    metrics = evaluate_model(
        results["model"],
        X_test,
        y_test,
    )

    print("Best parameters:", results["best_params"])
    print("CV accuracy:", results["cv_accuracy"])
    
    print("Test accuracy:", metrics["accuracy"])
    print("Confusion Matrix:", metrics["confusion_matrix"])