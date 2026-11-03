import numpy as np
import joblib
import pandas as pd

from src.config import MODELS_DIR, FEATURE_COLUMNS

from src.config import MODELS_DIR


def load_models():
    """Load both tuned classification models."""
    ovr_model = joblib.load(MODELS_DIR / "ovr_tuned_model.joblib")
    softmax_model = joblib.load(MODELS_DIR / "softmax_tuned_model.joblib")

    return ovr_model, softmax_model


def get_user_input():
    """Collect and validate the four flower measurements."""
    feature_names = [
        "Sepal length (cm)",
        "Sepal width (cm)",
        "Petal length (cm)",
        "Petal width (cm)"
    ]

    measurements = []

    print("\nEnter the flower measurements:")

    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))

                if not np.isfinite(value) or value <= 0:
                    print("Please enter a positive, finite number.")
                    continue

                measurements.append(value)
                break

            except ValueError:
                print("Invalid input. Please enter a number.")

    return pd.DataFrame([measurements], columns=FEATURE_COLUMNS)

def display_prediction(model, X, model_name):
    """Display predicted species and class probabilities."""
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    print(f"\n{model_name} Prediction")
    print("-" * 35)
    print(f"Predicted species: {prediction}")
    print("Class probabilities:")

    for species, probability in zip(model.classes_, probabilities):
        print(f"{species}: {probability * 100:.2f}%")


def predict_species():
    """Run predictions using both trained models."""
    ovr_model, softmax_model = load_models()

    X = get_user_input()

    print("\n========== PREDICTION RESULTS ==========")

    display_prediction(ovr_model, X, "One-vs-Rest Logistic Regression")
    display_prediction(softmax_model, X, "Softmax Logistic Regression")

    print("\n========================================")


if __name__ == "__main__":
    predict_species()
