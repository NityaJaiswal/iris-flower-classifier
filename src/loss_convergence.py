"""Compare OvR and multinomial logistic-regression loss convergence.

This module uses an explicit L-BFGS-B optimization diagnostic because
scikit-learn's LogisticRegression does not expose its complete per-iteration
loss history. The diagnostic uses the project's stratified training split,
training-only standardization, and the selected C value from GridSearchCV.

The recorded curves are diagnostic optimization histories, not a claim that
they are the internal iteration logs of scikit-learn's LogisticRegression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import FIGURES_DIR, MODELS_DIR
from src.preprocessing import split_data


BEST_C = 10.0
MAX_ITER = 1000


def _binary_objective(theta, X, y, regularization):
    """Mean binary cross-entropy plus L2 penalty (intercept unpenalized)."""
    weights = theta[:-1]
    bias = theta[-1]
    logits = X @ weights + bias
    loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
    loss += 0.5 * regularization * np.dot(weights, weights)
    probabilities = expit(logits)
    error = (probabilities - y) / len(y)
    grad_weights = X.T @ error + regularization * weights
    grad_bias = np.sum(error)
    return loss, np.r_[grad_weights, grad_bias]


def _softmax_objective(theta, X, y_encoded, class_count, regularization):
    """Multinomial cross-entropy plus L2 penalty (biases unpenalized)."""
    feature_count = X.shape[1]
    weights = theta[:feature_count * class_count].reshape(feature_count, class_count)
    biases = theta[feature_count * class_count:]
    logits = X @ weights + biases
    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    loss = -np.mean(log_probabilities[np.arange(len(y_encoded)), y_encoded])
    loss += 0.5 * regularization * np.sum(weights ** 2)

    probabilities = np.exp(log_probabilities)
    targets = np.eye(class_count)[y_encoded]
    error = (probabilities - targets) / len(y_encoded)
    grad_weights = X.T @ error + regularization * weights
    grad_biases = np.sum(error, axis=0)
    gradient = np.r_[grad_weights.ravel(), grad_biases]
    return loss, gradient


def _run_ovr(X, y_encoded, class_count, regularization):
    histories = []
    results = []
    for class_index in range(class_count):
        binary_y = (y_encoded == class_index).astype(float)
        initial = np.zeros(X.shape[1] + 1)
        history = []

        def objective(params):
            return _binary_objective(params, X, binary_y, regularization)

        history.append(objective(initial)[0])
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            callback=lambda params, h=history: h.append(objective(params)[0]),
            options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": 1e-9},
        )
        final_loss = objective(result.x)[0]
        if not np.isclose(history[-1], final_loss, rtol=1e-10, atol=1e-12):
            history.append(final_loss)
        histories.append(history)
        results.append(result)

    # The aggregate OvR objective is the sum of the K binary objectives.
    max_length = max(map(len, histories))
    aligned = [np.pad(h, (0, max_length - len(h)), mode="edge") for h in histories]
    aggregate_history = np.sum(np.vstack(aligned), axis=0)
    return aggregate_history, results


def _run_softmax(X, y_encoded, class_count, regularization):
    initial = np.zeros(X.shape[1] * class_count + class_count)
    history = []

    def objective(params):
        return _softmax_objective(params, X, y_encoded, class_count, regularization)

    history.append(objective(initial)[0])
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        callback=lambda params: history.append(objective(params)[0]),
        options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": 1e-9},
    )
    final_loss = objective(result.x)[0]
    if not np.isclose(history[-1], final_loss, rtol=1e-10, atol=1e-12):
        history.append(final_loss)
    return np.asarray(history), result


def compare_loss_convergence():
    """Fit diagnostic optimizers and save convergence plot plus CSV history."""
    X_train, _, y_train, _ = split_data()

    # Fit scaling on training data only to avoid data leakage.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)
    class_count = len(encoder.classes_)

    # C is the selected value reported by the project's hyperparameter search.
    # The mean-loss objective uses the L-BFGS LogisticRegression scaling
    # lambda = 1 / (C * n_samples).
    regularization = 1.0 / (BEST_C * len(y_encoded))

    ovr_history, ovr_results = _run_ovr(
        X_scaled, y_encoded, class_count, regularization
    )
    softmax_history, softmax_result = _run_softmax(
        X_scaled, y_encoded, class_count, regularization
    )

    max_iterations = max(len(ovr_history), len(softmax_history))
    history_df = pd.DataFrame({
        "Iteration": np.arange(max_iterations),
        "OvR_Aggregate_Binary_Cross_Entropy": np.pad(
            ovr_history, (0, max_iterations - len(ovr_history)), mode="edge"
        ),
        "Softmax_Categorical_Cross_Entropy": np.pad(
            softmax_history, (0, max_iterations - len(softmax_history)), mode="edge"
        ),
    })

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = MODELS_DIR / "loss_convergence_history.csv"
    figure_path = FIGURES_DIR / "ovr_vs_softmax_loss_convergence.png"
    history_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 6))
    plt.plot(
        np.arange(len(ovr_history)), ovr_history,
        marker="o", markersize=3, linewidth=2,
        label="One-vs-Rest (sum of binary losses)"
    )
    plt.plot(
        np.arange(len(softmax_history)), softmax_history,
        marker="s", markersize=3, linewidth=2,
        label="Multinomial (categorical loss)"
    )
    plt.title("OvR vs Softmax Logistic Regression: Loss Convergence")
    plt.xlabel("L-BFGS-B Iteration")
    plt.ylabel("Regularized Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("LOSS CONVERGENCE COMPARISON")
    print("-" * 55)
    print(f"Training samples: {len(y_train)}")
    print(f"Features: {X_scaled.shape[1]}")
    print(f"Classes: {list(encoder.classes_)}")
    print(f"C: {BEST_C} (mean-loss L2 strength = {regularization:.8f})")
    print(f"OvR iterations (per class): {[r.nit for r in ovr_results]}")
    print(f"OvR optimizer success (per class): {[r.success for r in ovr_results]}")
    print(f"OvR final aggregate loss: {ovr_history[-1]:.8f}")
    print(f"Softmax iterations: {softmax_result.nit}")
    print(f"Softmax optimizer success: {softmax_result.success}")
    print(f"Softmax final loss: {softmax_history[-1]:.8f}")
    print("Saved figure:", figure_path)
    print("Saved history:", csv_path)

    return history_df, figure_path, csv_path


if __name__ == "__main__":
    compare_loss_convergence()
