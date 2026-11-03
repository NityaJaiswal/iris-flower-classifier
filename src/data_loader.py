
import pandas as pd
from sklearn.datasets import load_iris

from src.config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN
)


def load_and_prepare_dataset():

    # Load the original Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # Rename feature columns
    df = df.rename(columns={
        "sepal length (cm)": "sepal_length_cm",
        "sepal width (cm)": "sepal_width_cm",
        "petal length (cm)": "petal_length_cm",
        "petal width (cm)": "petal_width_cm"
    })

    # Add unique sample IDs (1 to 150)
    df.insert(0, ID_COLUMN, range(1, len(df) + 1))

    # Convert numerical targets into species names
    species_mapping = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    df[TARGET_COLUMN] = df["target"].map(species_mapping)

    # Remove the original numerical target column
    df = df.drop(columns=["target"])

    # Arrange columns in the required order
    df = df[
        [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]
    ]

    return df


def save_dataset():

    # Prepare the dataset
    df = load_and_prepare_dataset()

    # Save as CSV
    df.to_csv(DATASET_PATH, index=False)

    print("Dataset saved successfully!")
    print("File location:", DATASET_PATH)
    print("Dataset shape:", df.shape)

    return df


if __name__ == "__main__":
    save_dataset()
