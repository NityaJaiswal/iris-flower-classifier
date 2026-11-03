# Iris Flower Classification: OvR vs Softmax Logistic Regression

A beginner-friendly yet technically strong machine learning project that compares two multiclass classification strategies — **One-vs-Rest (OvR)** and **Softmax (Multinomial) Logistic Regression** — using the original Iris dataset from Scikit-learn.

## Project Overview

This project builds, trains, tunes, and evaluates two Logistic Regression-based multiclass classifiers to predict the species of an Iris flower using four botanical measurements.

**Project objectives:**
- Understand multiclass classification.
- Implement OvR and Softmax Logistic Regression.
- Perform exploratory data analysis (EDA).
- Apply preprocessing without data leakage.
- Tune hyperparameters using cross-validation.
- Compare models using classification metrics and visualizations.
- Build an interactive prediction script.

## Dataset

The original Iris dataset from Scikit-learn contains:

| Property | Value |
|---|---:|
| Total samples | 150 |
| Features | 4 |
| Classes | 3 |
| Samples per class | 50 |
| Missing values | 0 |

**Input features:**
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

**Target classes:** Setosa, Versicolor, Virginica.

## Models Compared

1. **One-vs-Rest (OvR) Logistic Regression:** Trains one binary classifier per class.
2. **Softmax Logistic Regression:** Uses a multinomial approach to model class probabilities jointly.

Both models use StandardScaler within their pipelines.

## Methodology

1. Dataset loading and validation
2. Exploratory Data Analysis
3. Stratified train-test split (80:20)
4. Feature standardization
5. Model training
6. Hyperparameter tuning using 5-fold Stratified Cross-Validation
7. Model evaluation
8. Decision-boundary visualization
9. Interactive prediction
10. Automated testing

## Results

The following results were obtained during the project evaluation:

| Metric | OvR | Softmax |
|---|---:|---:|
| Best C | 10 | 10 |
| Best CV Macro F1 | 95.80% | 96.63% |
| Test Accuracy | 93.33% | 100.00% |
| Test Macro F1 | 93.33% | 100.00% |
| Macro ROC-AUC (OvR) | 99.33% | 100.00% |

**Observation:** Softmax Logistic Regression performed better on this particular held-out test split. The test set contains only 30 samples, so these results should not be interpreted as proof of perfect generalization.

## Project Structure

```text
iris-flower-classifier/
├── data/
│   └── iris_botanical_records.csv
├── documentation/
│   ├── project_report.md
│   └── report_figures/
├── models/
│   ├── ovr_model.joblib
│   ├── softmax_model.joblib
│   └── model_comparison.csv
├── notebooks/
│   └── iris_classification_analysis.ipynb
├── outputs/
│   ├── figures/
│   ├── diagrams/
│   └── reports/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── visualization.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── hyperparameter_tuning.py
│   ├── evaluate_model.py
│   ├── decision_boundary.py
│   ├── predict.py
│   └── generate_report.py
├── tests/
│   └── test_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd iris-flower-classifier
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Run these commands from the project root.

### Exploratory Data Analysis

```bash
python -m src.visualization
```

### Train the models

```bash
python -m src.train_model
```

### Hyperparameter tuning

```bash
python -m src.hyperparameter_tuning
```

### Evaluate models

```bash
python -m src.evaluate_model
```

### Generate decision boundaries

```bash
python -m src.decision_boundary
```

### Interactive prediction

```bash
python -m src.predict
```

### Run automated tests

```bash
python -m pytest -v
```

## Sample Prediction

Example input:

- Sepal length: 5.1 cm
- Sepal width: 3.5 cm
- Petal length: 1.4 cm
- Petal width: 0.2 cm

Both trained models predicted **Setosa** for this sample.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Pytest

## Limitations

- The dataset is small, containing only 150 samples.
- The test set contains 30 samples.
- Performance on this dataset does not guarantee performance on real-world flower measurements.
- Decision-boundary plots are visual aids and should not be confused with the held-out test evaluation.

## Documentation and Deliverables

The `documentation/` and `outputs/` folders contain the academic project report, figures, diagrams, and generated reports as they are finalized.

## Author

**Nitya Jaiswal**  
B.Tech — Computer Science and Engineering (Artificial Intelligence and Machine Learning)  
United Institute of Technology, Naini, Prayagraj

## License

This project is intended for academic and educational purposes.