
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

from src.config import MODELS_DIR, RANDOM_STATE
from src.preprocessing import split_data
from src.train_model import build_models


def tune_models():

    X_train, X_test, y_train, y_test = split_data()

    ovr_model, softmax_model = build_models()

    # Five-fold stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Hyperparameter grids
    ovr_grid = {
        "classifier__estimator__C": [
            0.01, 0.1, 1, 10, 100
        ]
    }

    softmax_grid = {
        "classifier__C": [
            0.01, 0.1, 1, 10, 100
        ]
    }

    # Tune OvR model
    print("Tuning OvR model...")

    ovr_search = GridSearchCV(
        estimator=ovr_model,
        param_grid=ovr_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )

    ovr_search.fit(X_train, y_train)

    # Tune Softmax model
    print("\nTuning Softmax model...")

    softmax_search = GridSearchCV(
        estimator=softmax_model,
        param_grid=softmax_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )

    softmax_search.fit(X_train, y_train)

    # Save optimized models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        ovr_search.best_estimator_,
        MODELS_DIR / "ovr_tuned_model.joblib"
    )

    joblib.dump(
        softmax_search.best_estimator_,
        MODELS_DIR / "softmax_tuned_model.joblib"
    )

    # Display actual results
    print("\nHYPERPARAMETER TUNING RESULTS")
    print("-" * 45)

    print("\nOvR Best Parameters:", ovr_search.best_params_)
    print("OvR Best CV Macro F1:", ovr_search.best_score_)

    print("\nSoftmax Best Parameters:", softmax_search.best_params_)
    print("Softmax Best CV Macro F1:", softmax_search.best_score_)

    return ovr_search, softmax_search


if __name__ == "__main__":
    tune_models()
