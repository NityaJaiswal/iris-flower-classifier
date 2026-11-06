"""Interpret learned logistic-regression coefficients for OvR and Softmax."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FEATURE_COLUMNS, FIGURES_DIR, MODELS_DIR


def extract_model_coefficients():
    """Return class-wise coefficients from both tuned, scaled pipelines."""
    model_paths = {
        "OvR": MODELS_DIR / "ovr_tuned_model.joblib",
        "Softmax": MODELS_DIR / "softmax_tuned_model.joblib",
    }
    rows = []

    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Tuned model not found: {model_path}. Run hyperparameter tuning first."
            )
        pipeline = joblib.load(model_path)
        classifier = pipeline.named_steps["classifier"]

        if model_name == "OvR":
            class_names = classifier.classes_
            coefficient_matrix = np.vstack([
                estimator.coef_.ravel() for estimator in classifier.estimators_
            ])
        else:
            class_names = classifier.classes_
            coefficient_matrix = classifier.coef_

        if coefficient_matrix.shape != (len(class_names), len(FEATURE_COLUMNS)):
            raise ValueError(
                f"Unexpected coefficient shape for {model_name}: "
                f"{coefficient_matrix.shape}"
            )

        for class_index, class_name in enumerate(class_names):
            for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
                coefficient = float(coefficient_matrix[class_index, feature_index])
                rows.append({
                    "Model": model_name,
                    "Species": str(class_name),
                    "Feature": feature_name,
                    "Coefficient": coefficient,
                    "AbsoluteCoefficient": abs(coefficient),
                })

    return pd.DataFrame(rows)


def analyze_feature_coefficients():
    """Save detailed coefficients, summary importance and diagnostic plots."""
    coefficient_df = extract_model_coefficients()
    summary_df = (
        coefficient_df.groupby(["Model", "Feature"], as_index=False)
        ["AbsoluteCoefficient"].mean()
        .rename(columns={"AbsoluteCoefficient": "MeanAbsoluteCoefficient"})
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = MODELS_DIR / "model_coefficients_by_species.csv"
    summary_path = MODELS_DIR / "feature_coefficient_importance.csv"
    comparison_path = FIGURES_DIR / "feature_coefficient_importance.png"
    heatmap_path = FIGURES_DIR / "model_coefficient_heatmaps.png"

    coefficient_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary_df,
        x="Feature",
        y="MeanAbsoluteCoefficient",
        hue="Model",
    )
    plt.title("Mean Absolute Logistic Regression Coefficients by Feature")
    plt.xlabel("Standardized Input Feature")
    plt.ylabel("Mean Absolute Coefficient")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for axis, model_name in zip(axes, ["OvR", "Softmax"]):
        matrix = (
            coefficient_df[coefficient_df["Model"] == model_name]
            .pivot(index="Species", columns="Feature", values="Coefficient")
            .reindex(columns=FEATURE_COLUMNS)
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            center=0,
            cmap="coolwarm",
            linewidths=0.5,
            ax=axis,
        )
        axis.set_title(f"{model_name} Coefficients by Species")
        axis.set_xlabel("Standardized Input Feature")
        axis.set_ylabel("Species")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("FEATURE COEFFICIENT ANALYSIS")
    print("-" * 55)
    print("Mean absolute coefficients (larger magnitude = stronger model influence):")
    print(summary_df.pivot(
        index="Feature", columns="Model", values="MeanAbsoluteCoefficient"
    ).round(4).to_string())
    print("\nSaved detailed coefficients:", detail_path)
    print("Saved summary:", summary_path)
    print("Saved comparison plot:", comparison_path)
    print("Saved class-wise heatmaps:", heatmap_path)

    return coefficient_df, summary_df, comparison_path, heatmap_path


if __name__ == "__main__":
    analyze_feature_coefficients()
