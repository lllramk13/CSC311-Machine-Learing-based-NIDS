from sklearn.neural_network import MLPClassifier
from model import load_data, train_model, evaluate_model, predict_model, plot_history

def build_model(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    alpha=0.0001,
    batch_size=32,
    max_iter=200,
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