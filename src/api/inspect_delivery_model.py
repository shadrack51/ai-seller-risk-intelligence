from pathlib import Path

import joblib


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


MODEL_PATH = (
    PROCESSED_DIR
    / "delivery_early_warning_model.joblib"
)


# =========================================================
# 2. LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


model = joblib.load(
    MODEL_PATH
)


print("=" * 70)
print("DELIVERY MODEL FEATURE SCHEMA")
print("=" * 70)


# =========================================================
# 3. CHECK MODEL TYPE
# =========================================================

print(
    f"\nModel type: "
    f"{type(model).__name__}"
)


# =========================================================
# 4. CHECK PIPELINE
# =========================================================

if hasattr(
    model,
    "steps",
):

    print(
        "\nPipeline steps:"
    )

    for step_name, step_object in model.steps:

        print(
            f"  - {step_name}: "
            f"{type(step_object).__name__}"
        )


# =========================================================
# 5. FIND FEATURE NAMES
# =========================================================

if not hasattr(
    model,
    "feature_names_in_",
):

    raise AttributeError(
        "\nThe loaded model does not expose "
        "feature_names_in_."
    )


feature_names = list(
    model.feature_names_in_
)


# =========================================================
# 6. DISPLAY FEATURE COUNT
# =========================================================

print(
    f"\nNumber of features used during training: "
    f"{len(feature_names)}"
)


# =========================================================
# 7. DISPLAY FEATURES
# =========================================================

print(
    "\nFeatures used during model training:"
)

print(
    "-" * 70
)

for number, feature in enumerate(
    feature_names,
    start=1,
):

    print(
        f"{number:02d}. {feature}"
    )


# =========================================================
# 8. FINAL SUMMARY
# =========================================================

print(
    "\n"
)

print(
    "=" * 70
)

print(
    "DELIVERY MODEL SCHEMA INSPECTION COMPLETE"
)

print(
    "=" * 70
)