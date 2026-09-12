from pathlib import Path

import joblib


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


# =========================================================
# 2. FIND MODEL ARTIFACTS
# =========================================================

model_files = sorted(
    PROCESSED_DIR.glob("*.joblib")
)


print("=" * 70)
print("MODEL ARTIFACT INVENTORY")
print("=" * 70)


if not model_files:
    print(
        "\nNo .joblib model artifacts were found."
    )

    raise FileNotFoundError(
        "No trained model artifacts are available."
    )


print(
    f"\nFound {len(model_files)} model artifact(s):"
)


# =========================================================
# 3. INSPECT EACH MODEL
# =========================================================

for file_path in model_files:

    print("\n")
    print("-" * 70)

    print(
        f"File: {file_path.name}"
    )

    print(
        f"Path: {file_path}"
    )

    try:

        model = joblib.load(
            file_path
        )

        print(
            f"Object type: {type(model)}"
        )

        # -------------------------------------------------
        # Check whether the object is a scikit-learn
        # pipeline.
        # -------------------------------------------------

        if hasattr(
            model,
            "steps",
        ):

            print(
                "Pipeline: YES"
            )

            print(
                "Pipeline steps:"
            )

            for step_name, step_object in model.steps:

                print(
                    f"  - {step_name}: "
                    f"{type(step_object).__name__}"
                )

        else:

            print(
                "Pipeline: NO"
            )

        # -------------------------------------------------
        # Check whether the model supports prediction.
        # -------------------------------------------------

        if hasattr(
            model,
            "predict",
        ):

            print(
                "predict(): YES"
            )

        else:

            print(
                "predict(): NO"
            )

        # -------------------------------------------------
        # Check whether probability prediction exists.
        # -------------------------------------------------

        if hasattr(
            model,
            "predict_proba",
        ):

            print(
                "predict_proba(): YES"
            )

        else:

            print(
                "predict_proba(): NO"
            )

    except Exception as exc:

        print(
            "LOAD FAILED"
        )

        print(
            f"Error: {exc}"
        )


# =========================================================
# 4. FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 70)

print(
    "MODEL ARTIFACT INVENTORY COMPLETE"
)

print(
    "=" * 70
)

print(
    """
The purpose of this step is to identify the exact
trained model artifacts that will be used by the
application layer.

We are not retraining any model here.
"""
)