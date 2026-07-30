from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, loguniform
from model import load_data, train_model, evaluate_model, predict_model, plot_history

def build_model(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    alpha=0.0001,
    batch_size=32,
    max_iter=500,
    early_stopping=False,
    random_state=42,
):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state,
    )

if __name__ == "__main__":
    print("I am running!")
    X_train, y_train, X_test, y_test = load_data()

    model = build_model()

    hyperparams = {
    "hidden_layer_sizes": [
        (50,),
        (100,),
        (100, 50),
        (200, 100),
    ],
    "activation": ["relu", "tanh"],
    "alpha": loguniform(1e-5, 1e-2),
    "learning_rate_init": loguniform(1e-4, 1e-2),
    "batch_size": randint(32, 256),
    "early_stopping": [True],
    }

    results = train_model(
        model=model,
        name="neural_network",
        X_train=X_train,
        y_train=y_train,
        hyperparams=hyperparams,
    )

    metrics = evaluate_model(results["model"], X_test, y_test)


    print("Best parameters:", results["best_params"])
    print("CV accuracy:", results["cv_accuracy"])

    print("Test accuracy:", metrics["accuracy"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    plot_history(results["model"])