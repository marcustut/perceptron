import numpy as np

DATA_DIR = "data/outputs"


def evaluate_letter_recognition():
    chunk_size = 26
    _targets = np.genfromtxt(
        f"{DATA_DIR}/letter_recognition_targets.csv", delimiter=","
    )
    _predictions = np.genfromtxt(
        f"{DATA_DIR}/letter_recognition_predictions.csv", delimiter=","
    )
    targets = []
    predictions = []
    get_letter = lambda index: chr(index + 65)

    for i in range(0, len(_targets), chunk_size):
        targets.append(get_letter(np.argmax(_targets[i : i + chunk_size])))
        predictions.append(get_letter(np.argmax(_predictions[i : i + chunk_size])))

    print(f"There are {len(targets)} targets and {len(predictions)} predictions")

    targets = np.array(targets)
    predictions = np.array(predictions)
    accuracy = np.sum(targets == predictions) / len(targets)

    print("Accuracy:", accuracy * 100, "%")


evaluate_letter_recognition()
