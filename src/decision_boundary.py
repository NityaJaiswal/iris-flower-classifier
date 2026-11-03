
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FIGURES_DIR,
    RANDOM_STATE
)


def plot_decision_boundary(feature_pair, model_type):

    # Load the complete dataset for visualization only
    df = pd.read_csv(DATASET_PATH)

    X = df[feature_pair]
    y = df[TARGET_COLUMN]

    # Build the selected two-feature model
    if model_type == "OvR":

        classifier = OneVsRestClassifier(
            LogisticRegression(
                C=10,
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )

    else:

        classifier = LogisticRegression(
            C=10,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE
        )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    # Train visualization model
    model.fit(X, y)

    # Create mesh grid
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    # Predict species for each grid point
    grid = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()],
        columns=feature_pair
    )

    predictions = model.predict(grid)

    # Convert predicted labels into numerical class indices
    class_mapping = {
        label: index
        for index, label in enumerate(model.classes_)
    }

    Z = np.array([
        class_mapping[label] for label in predictions
    ]).reshape(xx.shape)

    # Plot decision regions
    plt.figure(figsize=(9, 6))

    plt.contourf(
        xx, yy, Z,
        alpha=0.3,
        levels=np.arange(len(model.classes_) + 1) - 0.5,
        cmap="viridis"
    )

    # Plot original observations
    for species in model.classes_:

        subset = df[df[TARGET_COLUMN] == species]

        plt.scatter(
            subset[feature_pair[0]],
            subset[feature_pair[1]],
            label=species,
            edgecolors="black",
            s=35
        )

    plt.xlabel(feature_pair[0])
    plt.ylabel(feature_pair[1])

    plt.title(
        f"{model_type} Decision Boundary\n"
        f"{feature_pair[0]} vs {feature_pair[1]}"
    )

    plt.legend()
    plt.tight_layout()

    filename = (
        f"decision_boundary_{model_type.lower()}_"
        f"{feature_pair[0]}_{feature_pair[1]}.png"
    )

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Generated:", filename)


def generate_all_boundaries():

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Sepal feature pair
    sepal_features = [
        "sepal_length_cm",
        "sepal_width_cm"
    ]

    # Petal feature pair
    petal_features = [
        "petal_length_cm",
        "petal_width_cm"
    ]

    for features in [sepal_features, petal_features]:

        for model_type in ["OvR", "Softmax"]:

            plot_decision_boundary(
                features,
                model_type
            )

    print("\nAll four decision-boundary graphs generated!")


if __name__ == "__main__":
    generate_all_boundaries()
