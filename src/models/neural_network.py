from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, loguniform
from model import load_data, train_model, evaluate_model, predict_model, plot_history

def build_model(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    alpha=0.0001,
    batch_size=512,
    max_iter=100,
    early_stopping=True,
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
    X_train, y_train, X_test, y_test = load_data(
        datasets=["cse_cic_ids2018"],
        target="attack",
        sample_percentage=50,
    )

    model = build_model()

    hyperparams = {
    "hidden_layer_sizes": [
        (100,),
        (100, 50),
    ],
    "activation": [
        "relu",
    ],
    "alpha": [
        0.0001,
        0.001,
    ],
    "learning_rate_init": [
        0.001,
    ],
    "batch_size": [
        256,
        512,
    ],
    "early_stopping": [
        True,
    ],
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
    print("Class order:", metrics["classes"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    plot_history(results["model"])
