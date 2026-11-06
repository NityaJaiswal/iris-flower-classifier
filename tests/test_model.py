
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


def test_loss_convergence_diagnostic_generates_valid_outputs():
    from src.loss_convergence import compare_loss_convergence

    history, figure_path, csv_path = compare_loss_convergence()

    assert not history.empty
    assert history["OvR_Aggregate_Binary_Cross_Entropy"].notna().all()
    assert history["Softmax_Categorical_Cross_Entropy"].notna().all()
    assert history["OvR_Aggregate_Binary_Cross_Entropy"].iloc[-1] < history["OvR_Aggregate_Binary_Cross_Entropy"].iloc[0]
    assert history["Softmax_Categorical_Cross_Entropy"].iloc[-1] < history["Softmax_Categorical_Cross_Entropy"].iloc[0]
    assert figure_path.exists()
    assert csv_path.exists()


def test_feature_coefficient_analysis_outputs_valid_data():
    from src.feature_analysis import analyze_feature_coefficients

    detail, summary, comparison_path, heatmap_path = analyze_feature_coefficients()

    assert len(detail) == 24  # 2 models x 3 species x 4 features
    assert set(detail["Model"]) == {"OvR", "Softmax"}
    assert detail["Coefficient"].map(np.isfinite).all()
    assert len(summary) == 8  # 2 models x 4 features
    assert comparison_path.exists()
    assert heatmap_path.exists()


def test_eda_visualization_command_generates_species_distribution():
    from src.data_loader import load_and_prepare_dataset
    from src.visualization import save_species_distribution
    from src.config import FIGURES_DIR

    save_species_distribution(load_and_prepare_dataset())
    assert (FIGURES_DIR / "species_distribution.png").exists()
