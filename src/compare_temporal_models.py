
from pathlib import Path

import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = (
    PROCESSED_DIR / "final_temporal_model_comparison.csv"
)


# =========================================================
# 2. MODEL RESULTS
# =========================================================
#
# These are results from the models we already trained.
#
# We are NOT retraining anything here.
#
# The purpose of this script is to create one formal,
# auditable comparison table.
# =========================================================


results = [
    # -----------------------------------------------------
    # DELIVERY — RANDOM FOREST
    # -----------------------------------------------------

    {
        "target": "Delivery risk",
        "model": "Random Forest",
        "validation_roc_auc": 0.5476,
        "test_roc_auc": 0.5075,
        "validation_average_precision": 0.1771,
        "test_average_precision": 0.1202,
        "validation_threshold": 0.15,
        "validation_precision": 0.1748,
        "validation_recall": 0.7350,
        "validation_f1": 0.2824,
        "test_precision": 0.1325,
        "test_recall": 0.5780,
        "test_f1": 0.2156,
        "interpretation": (
            "Weak out-of-time discrimination"
        ),
    },

    # -----------------------------------------------------
    # DELIVERY — LOGISTIC REGRESSION
    # -----------------------------------------------------

    {
        "target": "Delivery risk",
        "model": "Logistic Regression",
        "validation_roc_auc": 0.5752,
        "test_roc_auc": 0.4595,
        "validation_average_precision": 0.1942,
        "test_average_precision": 0.1058,
        "validation_threshold": 0.16,
        "validation_precision": 0.1942,
        "validation_recall": 0.7158,
        "validation_f1": 0.3055,
        "test_precision": 0.1026,
        "test_recall": 0.3262,
        "test_f1": 0.1561,
        "interpretation": (
            "Poor out-of-time discrimination"
        ),
    },

    # -----------------------------------------------------
    # REVIEW — RANDOM FOREST
    # -----------------------------------------------------

    {
        "target": "Review risk",
        "model": "Random Forest",
        "validation_roc_auc": 0.5711,
        "test_roc_auc": 0.5879,
        "validation_average_precision": 0.2202,
        "test_average_precision": None,
        "validation_threshold": 0.10,
        "validation_precision": 0.1825,
        "validation_recall": 0.8206,
        "validation_f1": 0.2986,
        "test_precision": 0.1050,
        "test_recall": 0.7991,
        "test_f1": 0.1857,
        "interpretation": (
            "Moderate screening signal with poor precision"
        ),
    },

    # -----------------------------------------------------
    # REVIEW — LOGISTIC REGRESSION
    # -----------------------------------------------------

    {
        "target": "Review risk",
        "model": "Logistic Regression",
        "validation_roc_auc": 0.6214,
        "test_roc_auc": 0.6555,
        "validation_average_precision": 0.2396,
        "test_average_precision": 0.1639,
        "validation_threshold": 0.12,
        "validation_precision": 0.2017,
        "validation_recall": 0.8232,
        "validation_f1": 0.3240,
        "test_precision": 0.1225,
        "test_recall": 0.8348,
        "test_f1": 0.2137,
        "interpretation": (
            "Best available out-of-time screening model"
        ),
    },
]


# =========================================================
# 3. CREATE COMPARISON DATAFRAME
# =========================================================

comparison_df = pd.DataFrame(
    results
)


# =========================================================
# 4. CALCULATE MODEL CHANGES
# =========================================================

comparison_df[
    "generalization_gap"
] = (
    comparison_df["validation_roc_auc"]
    - comparison_df["test_roc_auc"]
)


comparison_df[
    "test_vs_validation_roc_auc"
] = (
    comparison_df["test_roc_auc"]
    - comparison_df["validation_roc_auc"]
)


# =========================================================
# 5. SELECT BEST MODEL FOR EACH TARGET
# =========================================================
#
# Primary selection criterion:
#
#     Highest TEST ROC-AUC
#
# Why?
#
# The test period represents the future period that the
# model did not use for training or threshold selection.
#
# We therefore use it as the strongest evidence of
# temporal generalization.
# =========================================================


selected_models = []


for target, group in comparison_df.groupby(
    "target"
):

    best_index = group[
        "test_roc_auc"
    ].idxmax()

    best_row = comparison_df.loc[
        best_index
    ]

    selected_models.append(
        {
            "target": target,
            "selected_model": best_row["model"],
            "test_roc_auc": best_row["test_roc_auc"],
            "validation_roc_auc": (
                best_row["validation_roc_auc"]
            ),
            "generalization_gap": (
                best_row["generalization_gap"]
            ),
            "reason": (
                "Highest out-of-time test ROC-AUC "
                "among evaluated models"
            ),
        }
    )


selection_df = pd.DataFrame(
    selected_models
)


# =========================================================
# 6. DISPLAY FULL COMPARISON
# =========================================================

print("=" * 80)
print("FORMAL TEMPORAL MODEL COMPARISON")
print("=" * 80)


print(
    "\nMODEL PERFORMANCE"
)

print(
    "-" * 80
)


display_columns = [
    "target",
    "model",
    "validation_roc_auc",
    "test_roc_auc",
    "validation_average_precision",
    "test_average_precision",
    "validation_threshold",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "test_precision",
    "test_recall",
    "test_f1",
    "generalization_gap",
]


print(
    comparison_df[
        display_columns
    ].to_string(
        index=False
    )
)


# =========================================================
# 7. MODEL SELECTION
# =========================================================

print(
    "\n"
)

print(
    "=" * 80
)

print(
    "FINAL MODEL SELECTION BY TARGET"
)

print(
    "=" * 80
)


for _, row in selection_df.iterrows():

    print(
        f"\nTarget: {row['target']}"
    )

    print(
        f"Selected model: {row['selected_model']}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{row['validation_roc_auc']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{row['test_roc_auc']:.4f}"
    )

    print(
        f"Validation → Test gap: "
        f"{row['generalization_gap']:.4f}"
    )

    print(
        f"Reason: {row['reason']}"
    )


# =========================================================
# 8. BUSINESS / ML POSTURE
# =========================================================

print(
    "\n"
)

print(
    "=" * 80
)

print(
    "MODEL POSTURE"
)

print(
    "=" * 80
)


print(
    """
DELIVERY RISK
-------------
Neither model demonstrates strong temporal
generalization.

The Random Forest test ROC-AUC is approximately
0.50, while Logistic Regression falls below 0.50.

Therefore delivery risk should be treated as a
monitoring / early-warning signal rather than a
high-confidence predictive model.


REVIEW RISK
-----------
Logistic Regression produces the strongest
out-of-time ranking performance.

Its test ROC-AUC of 0.6555 is higher than the
Random Forest's 0.5879.

However, the frozen operating threshold still
produces relatively low precision.

Therefore review risk should be treated as a
screening / prioritization model rather than
a definitive classifier.


OVERALL STRATEGY
----------------
Use target-specific modeling.

Do not force one algorithm to solve both targets.

Use Logistic Regression as the preferred review-risk
model.

Treat delivery-risk predictions cautiously because
the available temporal signal is weak.
"""
)


# =========================================================
# 9. SAVE COMPARISON
# =========================================================

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\nComparison table saved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\nFinal temporal model comparison complete."
)

