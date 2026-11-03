
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)


def load_data():

    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def split_data():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = split_data()

    print("DATA PREPROCESSING SUMMARY")
    print("-" * 40)

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())
