
import joblib
import pandas as pd
import numpy as np

from src.config import (
    DATASET_PATH,
    MODELS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)
from src.preprocessing import split_data


def test_dataset_shape_and_columns():
    df = pd.read_csv(DATASET_PATH)

    assert df.shape == (150, 6)
    assert all(col in df.columns for col in FEATURE_COLUMNS)
    assert TARGET_COLUMN in df.columns


def test_dataset_has_no_missing_values():
    df = pd.read_csv(DATASET_PATH)
    assert df.isnull().sum().sum() == 0


def test_dataset_has_three_balanced_classes():
    df = pd.read_csv(DATASET_PATH)

    counts = df[TARGET_COLUMN].value_counts()

    assert len(counts) == 3
    assert all(count == 50 for count in counts)


def test_train_test_split_shapes():
    X_train, X_test, y_train, y_test = split_data()

    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_saved_models_exist_and_load():
    ovr = joblib.load(MODELS_DIR / "ovr_tuned_model.joblib")
    softmax = joblib.load(MODELS_DIR / "softmax_tuned_model.joblib")

    assert ovr is not None
    assert softmax is not None


def test_models_predict_valid_species():
    ovr = joblib.load(MODELS_DIR / "ovr_tuned_model.joblib")
    softmax = joblib.load(MODELS_DIR / "softmax_tuned_model.joblib")

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=FEATURE_COLUMNS
    )

    valid_species = {"Setosa", "Versicolor", "Virginica"}

    assert ovr.predict(sample)[0] in valid_species
    assert softmax.predict(sample)[0] in valid_species


def test_prediction_probabilities_sum_to_one():
    ovr = joblib.load(MODELS_DIR / "ovr_tuned_model.joblib")
    softmax = joblib.load(MODELS_DIR / "softmax_tuned_model.joblib")

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=FEATURE_COLUMNS
    )

    for model in [ovr, softmax]:
        probabilities = model.predict_proba(sample)

        assert probabilities.shape == (1, 3)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.isclose(probabilities.sum(), 1.0)


def test_models_predict_same_class_for_setosa_sample():
    ovr = joblib.load(MODELS_DIR / "ovr_tuned_model.joblib")
    softmax = joblib.load(MODELS_DIR / "softmax_tuned_model.joblib")

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=FEATURE_COLUMNS
    )

    assert ovr.predict(sample)[0] == "Setosa"
    assert softmax.predict(sample)[0] == "Setosa"
