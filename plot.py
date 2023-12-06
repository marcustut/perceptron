import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/outputs"


def plot_sin_predictions():
    targets = np.genfromtxt(f"{DATA_DIR}/sin_targets.csv", delimiter=",")
    predictions = np.genfromtxt(f"{DATA_DIR}/sin_predictions.csv", delimiter=",")

    plt.title("Sin(x) Predictions vs Targets")
    plt.plot(targets, label="Targets")
    plt.plot(predictions, label="Predictions")
    plt.legend()
    plt.savefig("images/sin_predictions.png")


plot_sin_predictions()
