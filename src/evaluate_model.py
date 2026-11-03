
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize

from src.config import (
    MODELS_DIR,
    FIGURES_DIR
)
from src.preprocessing import split_data


def evaluate_models():

    # Load the same untouched test set
    X_train, X_test, y_train, y_test = split_data()

    # Load tuned models
    ovr_model = joblib.load(
        MODELS_DIR / "ovr_tuned_model.joblib"
    )

    softmax_model = joblib.load(
        MODELS_DIR / "softmax_tuned_model.joblib"
    )

    models = {
        "OvR": ovr_model,
        "Softmax": softmax_model
    }

    results = []

    # Evaluate both models
    for name, model in models.items():

        print("\n" + "=" * 55)
        print(f"{name} MODEL EVALUATION")
        print("=" * 55)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision_macro = precision_score(
            y_test, y_pred, average="macro"
        )

        precision_weighted = precision_score(
            y_test, y_pred, average="weighted"
        )

        recall_macro = recall_score(
            y_test, y_pred, average="macro"
        )

        recall_weighted = recall_score(
            y_test, y_pred, average="weighted"
        )

        f1_macro = f1_score(
            y_test, y_pred, average="macro"
        )

        f1_weighted = f1_score(
            y_test, y_pred, average="weighted"
        )

        roc_auc = roc_auc_score(
            y_test,
            y_proba,
            labels=model.classes_,
            multi_class="ovr",
            average="macro"
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Precision (Weighted): {precision_weighted:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        print(f"Recall (Weighted): {recall_weighted:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"F1 (Weighted): {f1_weighted:.4f}")
        print(f"ROC-AUC (Macro OvR): {roc_auc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=model.classes_
        )

        plt.figure(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_
        )

        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Species")
        plt.ylabel("Actual Species")
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"{name.lower()}_confusion_matrix.png",
            dpi=300
        )

        plt.close()

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision Macro": precision_macro,
            "Precision Weighted": precision_weighted,
            "Recall Macro": recall_macro,
            "Recall Weighted": recall_weighted,
            "F1 Macro": f1_macro,
            "F1 Weighted": f1_weighted,
            "ROC-AUC Macro OvR": roc_auc
        })

    # Per-class ROC and Precision-Recall curves
    classes = ovr_model.classes_
    y_binary = label_binarize(y_test, classes=classes)

    plt.figure(figsize=(8, 6))

    for name, model in models.items():

        probabilities = model.predict_proba(X_test)

        for i, species in enumerate(classes):

            fpr, tpr, _ = roc_curve(
                y_binary[:, i],
                probabilities[:, i]
            )

            plt.plot(
                fpr,
                tpr,
                label=f"{name} - {species}"
            )

    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("Multiclass ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "multiclass_roc_curves.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(8, 6))

    for name, model in models.items():

        probabilities = model.predict_proba(X_test)

        for i, species in enumerate(classes):

            precision, recall, _ = precision_recall_curve(
                y_binary[:, i],
                probabilities[:, i]
            )

            plt.plot(
                recall,
                precision,
                label=f"{name} - {species}"
            )

    plt.title("Multiclass Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "multiclass_precision_recall.png",
        dpi=300
    )

    plt.close()

    # Save comparison table
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        MODELS_DIR / "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 55)
    print("FINAL MODEL COMPARISON")
    print("=" * 55)

    print(results_df.to_string(index=False))

    print("\nEvaluation completed successfully!")
    print("Comparison saved in models/model_comparison.csv")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    evaluate_models()
