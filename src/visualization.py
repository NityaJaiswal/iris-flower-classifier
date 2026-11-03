
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FIGURES_DIR
)


# Consistent appearance for all project graphs
sns.set_theme(style="whitegrid", palette="Set2")

# Ensure output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_histograms(df):
    """Generate feature distribution histograms."""

    for feature in FEATURE_COLUMNS:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            hue=TARGET_COLUMN,
            kde=True,
            bins=15
        )

        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"histogram_{feature}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


def save_boxplots(df):
    """Generate species-wise box plots."""

    for feature in FEATURE_COLUMNS:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=TARGET_COLUMN,
            y=feature
        )

        plt.title(f"Box Plot of {feature} by Species")
        plt.xlabel("Species")
        plt.ylabel(feature)
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"boxplot_{feature}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


def save_pairplot(df):
    """Generate pairwise feature relationships."""

    plot = sns.pairplot(
        df,
        vars=FEATURE_COLUMNS,
        hue=TARGET_COLUMN,
        diag_kind="hist"
    )

    plot.figure.suptitle(
        "Pairwise Relationships of Iris Features",
        y=1.02
    )

    plot.savefig(
        FIGURES_DIR / "pairplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")


def save_correlation_heatmap(df):
    """Generate feature correlation heatmap."""

    plt.figure(figsize=(9, 7))

    correlation = df[FEATURE_COLUMNS].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap of Iris Features")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_species_distribution(df):
    """Generate species count plot."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=TARGET_COLUMN
    )

    plt.title("Distribution of Iris Species")
    plt.xlabel("Species")
    plt.ylabel("Number of Samples")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "species_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def generate_all_visualizations(df):
    """Generate all EDA visualizations."""

    save_histograms(df)
    save_boxplots(df)
    save_pairplot(df)
    save_correlation_heatmap(df)
    save_species_distribution(df)

    print("All EDA visualizations generated successfully!")
    print("Saved in:", FIGURES_DIR)
