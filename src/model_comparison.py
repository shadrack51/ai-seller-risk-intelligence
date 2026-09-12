from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

BASELINE_RESULTS_PATH = (
    PROCESSED_DATA_DIR / "baseline_model_results.csv"
)

RANDOM_FOREST_RESULTS_PATH = (
    PROCESSED_DATA_DIR / "random_forest_results.csv"
)

OUTPUT_PATH = (
    PROCESSED_DATA_DIR / "model_comparison.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_model_results(file_path, model_name):
    """
    Load evaluation results for one model.

    The expected CSV format is:

        metric,value

    or:

        metric,value,model
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            f"Results file is empty:\n{file_path}"
        )

    print(f"{model_name} results loaded.")

    return df


def clean_metric_results(df, model_name):
    """
    Clean and standardize model evaluation results.

    Handles cases where:
    - metric names contain spaces
    - metric names have different capitalization
    - CSV files contain an existing model column
    - blank rows exist
    """

    df = df.copy()

    print(f"\nCleaning results for {model_name}...")

    # --------------------------------------------------------
    # Standardize column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    print("Columns found:", list(df.columns))

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    if "metric" not in df.columns:
        raise ValueError(
            f"{model_name} results must contain a 'metric' column."
        )

    if "value" not in df.columns:
        raise ValueError(
            f"{model_name} results must contain a 'value' column."
        )

    # --------------------------------------------------------
    # Keep only useful columns
    # --------------------------------------------------------

    df = df[["metric", "value"]].copy()

    # --------------------------------------------------------
    # Clean metric names
    # --------------------------------------------------------

    df["metric"] = (
        df["metric"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Remove accidental blank / invalid metric rows
    df = df[
        df["metric"].notna()
        & (df["metric"] != "")
        & (df["metric"] != "nan")
    ].copy()

    # --------------------------------------------------------
    # Convert metric values to numeric
    # --------------------------------------------------------

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    # Remove rows where value could not be converted
    df = df[df["value"].notna()].copy()

    # --------------------------------------------------------
    # Keep only required metrics
    # --------------------------------------------------------

    df = df[
        df["metric"].isin(REQUIRED_METRICS)
    ].copy()

    # --------------------------------------------------------
    # Check duplicate metrics
    # --------------------------------------------------------

    duplicate_metrics = (
        df["metric"]
        .duplicated()
        .sum()
    )

    if duplicate_metrics > 0:
        raise ValueError(
            f"{model_name} contains duplicate metric names."
        )

    # --------------------------------------------------------
    # Validate required metrics
    # --------------------------------------------------------

    missing_metrics = [
        metric
        for metric in REQUIRED_METRICS
        if metric not in df["metric"].values
    ]

    if missing_metrics:
        raise ValueError(
            f"{model_name} is missing required metrics:\n"
            + "\n".join(missing_metrics)
        )

    print(
        f"{model_name} contains all required evaluation metrics."
    )

    return df


def reshape_model_results(df, model_name):
    """
    Convert metric/value rows into a single model row.

    Example:

        metric       value
        accuracy     0.92
        precision    0.91
        recall       0.81
        f1_score     0.86
        roc_auc      0.95

    becomes:

        model                 accuracy  precision  recall ...
        Logistic Regression   0.92      0.91       0.81
    """

    print(
        f"\nConverting {model_name} "
        "metric/value format into one model row."
    )

    # --------------------------------------------------------
    # Create dictionary instead of using pandas pivot().
    #
    # This avoids the previous:
    # KeyError: None
    #
    # and is simpler for this type of dataset.
    # --------------------------------------------------------

    metric_values = dict(
        zip(
            df["metric"],
            df["value"]
        )
    )

    row = {
        "model": model_name
    }

    for metric in REQUIRED_METRICS:
        row[metric] = metric_values[metric]

    return pd.DataFrame([row])


def validate_model_comparison(comparison_df):
    """
    Validate the final comparison table.
    """

    print(
        "\n======================================================================"
    )
    print("FINAL COMPARISON VALIDATION")
    print(
        "======================================================================"
    )

    # --------------------------------------------------------
    # Validate model count
    # --------------------------------------------------------

    if comparison_df["model"].duplicated().any():
        raise ValueError(
            "Duplicate models found in comparison table."
        )

    # --------------------------------------------------------
    # Validate metrics
    # --------------------------------------------------------

    for metric in REQUIRED_METRICS:

        if metric not in comparison_df.columns:
            raise ValueError(
                f"Missing metric column: {metric}"
            )

        missing_values = comparison_df[metric].isna().sum()

        if missing_values > 0:
            raise ValueError(
                f"Missing values found in metric: {metric}"
            )

    # --------------------------------------------------------
    # Validate metric ranges
    # --------------------------------------------------------

    for metric in REQUIRED_METRICS:

        invalid_values = (
            (comparison_df[metric] < 0)
            | (comparison_df[metric] > 1)
        ).sum()

        if invalid_values > 0:
            raise ValueError(
                f"Invalid values found in {metric}."
            )

    print("Model comparison validation -> PASS")


def calculate_winner(comparison_df):
    """
    Determine the best model for each metric.

    For every metric, the model with the highest score wins.
    """

    print(
        "\n======================================================================"
    )
    print("METRIC WINNERS")
    print(
        "======================================================================"
    )

    winner_rows = []

    for metric in REQUIRED_METRICS:

        best_index = comparison_df[metric].idxmax()

        best_model = comparison_df.loc[
            best_index,
            "model"
        ]

        best_value = comparison_df.loc[
            best_index,
            metric
        ]

        winner_rows.append(
            {
                "metric": metric,
                "best_model": best_model,
                "score": best_value,
            }
        )

        print(
            f"{metric:<12} -> "
            f"{best_model:<22} "
            f"{best_value:.4f}"
        )

    return pd.DataFrame(winner_rows)


def calculate_overall_winner(comparison_df):
    """
    Calculate the average of the five evaluation metrics.

    This is a simple summary score, not a replacement for
    looking at each metric individually.
    """

    comparison_df = comparison_df.copy()

    comparison_df["average_metric_score"] = (
        comparison_df[REQUIRED_METRICS]
        .mean(axis=1)
    )

    best_index = comparison_df[
        "average_metric_score"
    ].idxmax()

    best_model = comparison_df.loc[
        best_index,
        "model"
    ]

    best_score = comparison_df.loc[
        best_index,
        "average_metric_score"
    ]

    return comparison_df, best_model, best_score


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("MODEL COMPARISON ANALYSIS")
    print("=" * 70)

    # ========================================================
    # 1. LOAD RESULTS
    # ========================================================

    print("\nLoading model evaluation results...")

    baseline_results = load_model_results(
        BASELINE_RESULTS_PATH,
        "Logistic Regression"
    )

    random_forest_results = load_model_results(
        RANDOM_FOREST_RESULTS_PATH,
        "Random Forest"
    )

    # ========================================================
    # 2. DISPLAY RAW RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("RAW MODEL RESULTS")
    print("=" * 70)

    print("\nLogistic Regression:")
    print(baseline_results)

    print("\nRandom Forest:")
    print(random_forest_results)

    # ========================================================
    # 3. CLEAN RESULTS
    # ========================================================

    baseline_clean = clean_metric_results(
        baseline_results,
        "Logistic Regression"
    )

    random_forest_clean = clean_metric_results(
        random_forest_results,
        "Random Forest"
    )

    # ========================================================
    # 4. RESHAPE RESULTS
    # ========================================================

    baseline_wide = reshape_model_results(
        baseline_clean,
        "Logistic Regression"
    )

    random_forest_wide = reshape_model_results(
        random_forest_clean,
        "Random Forest"
    )

    # ========================================================
    # 5. BUILD COMPARISON TABLE
    # ========================================================

    print("\n" + "=" * 70)
    print("BUILDING MODEL COMPARISON TABLE")
    print("=" * 70)

    comparison_table = pd.concat(
        [
            baseline_wide,
            random_forest_wide,
        ],
        ignore_index=True
    )

    # ========================================================
    # 6. VALIDATE COMPARISON TABLE
    # ========================================================

    validate_model_comparison(
        comparison_table
    )

    # ========================================================
    # 7. DISPLAY COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    display_table = comparison_table.copy()

    for metric in REQUIRED_METRICS:
        display_table[metric] = (
            display_table[metric] * 100
        ).round(2)

    print(
        display_table.to_string(
            index=False
        )
    )

    # ========================================================
    # 8. CALCULATE METRIC WINNERS
    # ========================================================

    winners = calculate_winner(
        comparison_table
    )

    # ========================================================
    # 9. CALCULATE OVERALL SCORE
    # ========================================================

    (
        comparison_with_score,
        overall_winner,
        overall_score,
    ) = calculate_overall_winner(
        comparison_table
    )

    print("\n" + "=" * 70)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 70)

    print(
        comparison_with_score[
            [
                "model",
                "average_metric_score",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nOverall best model: {overall_winner}"
    )

    print(
        f"Average metric score: "
        f"{overall_score:.4f}"
    )

    # ========================================================
    # 10. MODEL-BY-MODEL INTERPRETATION
    # ========================================================

    print("\n" + "=" * 70)
    print("BUSINESS INTERPRETATION")
    print("=" * 70)

    logistic_row = comparison_table[
        comparison_table["model"]
        == "Logistic Regression"
    ].iloc[0]

    forest_row = comparison_table[
        comparison_table["model"]
        == "Random Forest"
    ].iloc[0]

    print("\nLogistic Regression:")

    print(
        f"- Accuracy: "
        f"{logistic_row['accuracy']:.4f}"
    )

    print(
        f"- Precision: "
        f"{logistic_row['precision']:.4f}"
    )

    print(
        f"- Recall: "
        f"{logistic_row['recall']:.4f}"
    )

    print(
        f"- F1 Score: "
        f"{logistic_row['f1_score']:.4f}"
    )

    print(
        f"- ROC-AUC: "
        f"{logistic_row['roc_auc']:.4f}"
    )

    print("\nRandom Forest:")

    print(
        f"- Accuracy: "
        f"{forest_row['accuracy']:.4f}"
    )

    print(
        f"- Precision: "
        f"{forest_row['precision']:.4f}"
    )

    print(
        f"- Recall: "
        f"{forest_row['recall']:.4f}"
    )

    print(
        f"- F1 Score: "
        f"{forest_row['f1_score']:.4f}"
    )

    print(
        f"- ROC-AUC: "
        f"{forest_row['roc_auc']:.4f}"
    )

    # ========================================================
    # 11. EXPLAIN MODEL SELECTION
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    print(
        "\nThe Random Forest model should be preferred "
        "for the current project because it performs "
        "better across all five evaluation metrics."
    )

    print(
        "\nHowever, the final model decision should not "
        "be based only on accuracy."
    )

    print(
        "For this business-risk problem, recall is especially "
        "important because false negatives represent sellers "
        "who are actually at risk but were not identified."
    )

    print(
        f"\nRandom Forest recall: "
        f"{forest_row['recall']:.4f}"
    )

    print(
        f"Logistic Regression recall: "
        f"{logistic_row['recall']:.4f}"
    )

    recall_improvement = (
        forest_row["recall"]
        - logistic_row["recall"]
    )

    print(
        f"Recall improvement: "
        f"{recall_improvement:.4f}"
    )

    # ========================================================
    # 12. SAVE RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVING MODEL COMPARISON RESULTS")
    print("=" * 70)

    comparison_table.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nModel comparison saved to:\n"
        f"{OUTPUT_PATH}"
    )

    # ========================================================
    # 13. SAVE WINNERS
    # ========================================================

    winners_path = (
        PROCESSED_DATA_DIR
        / "model_metric_winners.csv"
    )

    winners.to_csv(
        winners_path,
        index=False
    )

    print(
        f"\nMetric winners saved to:\n"
        f"{winners_path}"
    )

    # ========================================================
    # 14. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 70)

    print(
        f"\nModels compared: "
        f"{comparison_table['model'].nunique()}"
    )

    print(
        f"Metrics compared: "
        f"{len(REQUIRED_METRICS)}"
    )

    print(
        f"Overall best model: "
        f"{overall_winner}"
    )

    print(
        f"Overall average score: "
        f"{overall_score:.4f}"
    )

    print("\nMetric winners:")

    for _, row in winners.iterrows():

        print(
            f" - {row['metric']}: "
            f"{row['best_model']}"
        )

    print(
        "\nThe test set remains untouched by model "
        "training and is used only for evaluation."
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()