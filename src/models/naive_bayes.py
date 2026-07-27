from model import load_data, train_model, evaluate_model, predict_model, plot_history
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