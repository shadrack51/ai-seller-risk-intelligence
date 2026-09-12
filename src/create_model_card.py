from pathlib import Path

import json
import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COMPARISON_FILE = (
    PROCESSED_DIR / "final_temporal_model_comparison.csv"
)

MODEL_CARD_CSV = (
    PROCESSED_DIR / "model_decision_record.csv"
)

MODEL_CARD_JSON = (
    PROCESSED_DIR / "model_card.json"
)


# =========================================================
# 2. LOAD MODEL COMPARISON RESULTS
# =========================================================

if not COMPARISON_FILE.exists():
    raise FileNotFoundError(
        f"Model comparison file not found:\n"
        f"{COMPARISON_FILE}\n\n"
        f"Run compare_temporal_models.py first."
    )


comparison_df = pd.read_csv(
    COMPARISON_FILE
)


print("=" * 70)
print("FINAL ML MODEL DECISION RECORD")
print("=" * 70)


# =========================================================
# 3. SELECT BEST MODEL FOR EACH TARGET
# =========================================================

delivery_models = comparison_df[
    comparison_df["target"] == "Delivery risk"
]

review_models = comparison_df[
    comparison_df["target"] == "Review risk"
]


delivery_best = delivery_models.loc[
    delivery_models["test_roc_auc"].idxmax()
]

review_best = review_models.loc[
    review_models["test_roc_auc"].idxmax()
]


# =========================================================
# 4. FINAL MODEL DECISIONS
# =========================================================

model_decisions = {

    "delivery": {
        "target": "Future delivery risk",

        "selected_model": (
            delivery_best["model"]
        ),

        "validation_roc_auc": float(
            delivery_best["validation_roc_auc"]
        ),

        "test_roc_auc": float(
            delivery_best["test_roc_auc"]
        ),

        "validation_threshold": float(
            delivery_best["validation_threshold"]
        ),

        "test_precision": float(
            delivery_best["test_precision"]
        ),

        "test_recall": float(
            delivery_best["test_recall"]
        ),

        "test_f1": float(
            delivery_best["test_f1"]
        ),

        "role": (
            "Monitoring and early-warning signal"
        ),

        "confidence": "Low",

        "production_use": (
            "Cautious operational monitoring only"
        ),

        "not_for": (
            "High-confidence automated decisions "
            "or guaranteed predictions"
        ),

        "reason": (
            "Random Forest had the highest test ROC-AUC "
            "among the evaluated delivery models, but "
            "performance remained close to random ranking."
        ),
    },


    "review": {
        "target": "Future review risk",

        "selected_model": (
            review_best["model"]
        ),

        "validation_roc_auc": float(
            review_best["validation_roc_auc"]
        ),

        "test_roc_auc": float(
            review_best["test_roc_auc"]
        ),

        "validation_threshold": float(
            review_best["validation_threshold"]
        ),

        "test_precision": float(
            review_best["test_precision"]
        ),

        "test_recall": float(
            review_best["test_recall"]
        ),

        "test_f1": float(
            review_best["test_f1"]
        ),

        "role": (
            "Screening and prioritization signal"
        ),

        "confidence": "Moderate-low",

        "production_use": (
            "Prioritize sellers for review and investigation"
        ),

        "not_for": (
            "Automatic punitive decisions "
            "or guaranteed predictions"
        ),

        "reason": (
            "Logistic Regression achieved the strongest "
            "out-of-time ranking performance among the "
            "evaluated review-risk models."
        ),
    },
}


# =========================================================
# 5. DATASET AND EVALUATION CONTEXT
# =========================================================

evaluation_context = {

    "temporal_observations": 10666,

    "unique_sellers": 1684,

    "snapshots": 20,

    "train_observations": 6046,

    "validation_observations": 2272,

    "test_observations": 2348,

    "evaluation_method": (
        "Temporal train/validation/test evaluation"
    ),

    "threshold_selection": (
        "Validation data only"
    ),

    "test_usage": (
        "Final out-of-time evaluation only"
    ),

    "calibration_status": (
        "Scores should not be interpreted as "
        "well-calibrated literal probabilities"
    ),
}


# =========================================================
# 6. LIMITATIONS
# =========================================================

limitations = [

    "Delivery-risk temporal discrimination is weak.",

    (
        "Review-risk performance is stronger, "
        "but precision remains relatively low."
    ),

    (
        "Temporal changes in the data may cause "
        "future model performance to drift."
    ),

    (
        "Model coefficients and SHAP values describe "
        "model behavior and associations, not causation."
    ),

    (
        "Predictions should be treated as screening "
        "or monitoring signals rather than guarantees."
    ),

    (
        "The final test period was not used for "
        "training or threshold selection."
    ),
]


# =========================================================
# 7. BUSINESS ACTIONS
# =========================================================

business_actions = {

    "delivery": {

        "purpose": (
            "Identify sellers that may require "
            "delivery-performance monitoring."
        ),

        "recommended_action": (
            "Investigate delivery trends, late-order "
            "behavior, and operational performance."
        ),

        "automation_level": (
            "Human review required"
        ),
    },


    "review": {

        "purpose": (
            "Prioritize sellers whose patterns are "
            "associated with future review risk."
        ),

        "recommended_action": (
            "Review customer feedback, seller experience "
            "indicators, and recent operational changes."
        ),

        "automation_level": (
            "Human review required"
        ),
    },
}


# =========================================================
# 8. BUILD DECISION TABLE
# =========================================================

decision_rows = []


for target_key, decision in model_decisions.items():

    action = business_actions[
        target_key
    ]

    decision_rows.append(
        {
            "target": decision["target"],

            "selected_model": (
                decision["selected_model"]
            ),

            "validation_roc_auc": (
                decision["validation_roc_auc"]
            ),

            "test_roc_auc": (
                decision["test_roc_auc"]
            ),

            "validation_threshold": (
                decision["validation_threshold"]
            ),

            "test_precision": (
                decision["test_precision"]
            ),

            "test_recall": (
                decision["test_recall"]
            ),

            "test_f1": (
                decision["test_f1"]
            ),

            "role": decision["role"],

            "confidence": decision["confidence"],

            "production_use": (
                decision["production_use"]
            ),

            "recommended_action": (
                action["recommended_action"]
            ),

            "automation_level": (
                action["automation_level"]
            ),
        }
    )


decision_df = pd.DataFrame(
    decision_rows
)


# =========================================================
# 9. PRINT FINAL MODEL DECISIONS
# =========================================================

print(
    "\nFINAL MODEL DECISIONS"
)

print(
    "-" * 70
)

print(
    decision_df.to_string(
        index=False
    )
)


# =========================================================
# 10. PRINT LIMITATIONS
# =========================================================

print(
    "\nMODEL LIMITATIONS"
)

print(
    "-" * 70
)

for number, limitation in enumerate(
    limitations,
    start=1,
):

    print(
        f"{number}. {limitation}"
    )


# =========================================================
# 11. SAVE CSV DECISION RECORD
# =========================================================

decision_df.to_csv(
    MODEL_CARD_CSV,
    index=False
)


# =========================================================
# 12. BUILD MACHINE-READABLE MODEL CARD
# =========================================================

model_card = {

    "project": {
        "name": (
            "AI Seller Risk Intelligence System"
        ),

        "purpose": (
            "Identify seller patterns associated "
            "with future delivery and review risk."
        ),
    },

    "evaluation": evaluation_context,

    "models": model_decisions,

    "business_actions": business_actions,

    "limitations": limitations,

    "governance": {

        "human_review_required": True,

        "automatic_punitive_decisions": False,

        "literal_probability_claims": False,

        "test_set_used_for_training": False,

        "test_set_used_for_threshold_selection": False,
    },
}


# =========================================================
# 13. SAVE JSON MODEL CARD
# =========================================================

with open(
    MODEL_CARD_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        model_card,
        file,
        indent=4,
    )


# =========================================================
# 14. FINAL OUTPUT
# =========================================================

print(
    "\n"
)

print(
    "=" * 70
)

print(
    "MODEL DECISION RECORD COMPLETE"
)

print(
    "=" * 70
)

print(
    f"\nCSV saved to:"
)

print(
    MODEL_CARD_CSV
)

print(
    f"\nJSON model card saved to:"
)

print(
    MODEL_CARD_JSON
)

print(
    "\nThe ML model decisions are now formally documented."
)