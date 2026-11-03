
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, RANDOM_STATE
from src.preprocessing import split_data


def build_models():

    # Model A: One-vs-Rest Logistic Regression
    ovr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        ))
    ])

    # Model B: Multinomial / Softmax Logistic Regression
    softmax_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])

    return ovr_model, softmax_model


def train_models():

    # Obtain training and testing data
    X_train, X_test, y_train, y_test = split_data()

    # Build both classifiers
    ovr_model, softmax_model = build_models()

    # Train Model A
    print("Training OvR Logistic Regression...")
    ovr_model.fit(X_train, y_train)

    # Train Model B
    print("Training Softmax Logistic Regression...")
    softmax_model.fit(X_train, y_train)

    # Create model directory if necessary
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained models
    joblib.dump(
        ovr_model,
        MODELS_DIR / "ovr_model.joblib"
    )

    joblib.dump(
        softmax_model,
        MODELS_DIR / "softmax_model.joblib"
    )

    print("\nBoth models trained successfully!")
    print("OvR model:", MODELS_DIR / "ovr_model.joblib")
    print("Softmax model:", MODELS_DIR / "softmax_model.joblib")

    return ovr_model, softmax_model, X_test, y_test


if __name__ == "__main__":
    train_models()
