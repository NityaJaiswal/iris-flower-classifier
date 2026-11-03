
from pathlib import Path

# ==========================================
# PROJECT DIRECTORY CONFIGURATION
# ==========================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Output subdirectories
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
DIAGRAMS_DIR = OUTPUTS_DIR / "diagrams"

# Dataset file path
DATASET_PATH = DATA_DIR / "iris_botanical_records.csv"


# ==========================================
# MACHINE LEARNING CONFIGURATION
# ==========================================

# Reproducibility
RANDOM_STATE = 42

# Train-test split ratio
TEST_SIZE = 0.20


# ==========================================
# DATASET COLUMN CONFIGURATION
# ==========================================

# Independent variables (features)
FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm"
]

# Dependent variable (target)
TARGET_COLUMN = "species_category"

# Unique identification column
ID_COLUMN = "sample_id"
