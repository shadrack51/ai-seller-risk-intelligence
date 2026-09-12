"""
Target-Specific SHAP Explainability Analysis

Purpose
-------
Explain the two target-specific Random Forest models:

    1. future_delivery_risk
    2. future_review_risk

The analysis provides:

    - global mean absolute SHAP importance
    - directional SHAP contribution
    - validation-set explanations
    - selected local seller explanations
    - out-of-time test explanations as diagnostics

Important
---------
No model is retrained.

No threshold is changed.

No model or hyperparameter is selected using SHAP results.

Validation data is the primary explanation population.

The final test set is used only as an out-of-time diagnostic.

Temporal grain
--------------
The temporal datasets are observed at:

    seller_id + snapshot_date

Therefore, seller_id is expected to repeat across
multiple snapshots.

Only duplicate seller/snapshot combinations are invalid.

Inputs
-------
data/processed/delivery_risk_model.joblib
data/processed/review_risk_model.joblib

data/processed/target_delivery_validation.csv
data/processed/target_delivery_test.csv

data/processed/target_review_validation.csv
data/processed/target_review_test.csv

Outputs
-------
data/processed/delivery_shap_global.csv
data/processed/review_shap_global.csv

data/processed/delivery_shap_validation_local.csv
data/processed/review_shap_validation_local.csv

data/processed/delivery_shap_test_local.csv
data/processed/review_shap_test_local.csv
"""


from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# MODEL FILES
# ============================================================

DELIVERY_MODEL_FILE = (
    DATA_DIR
    / "delivery_risk_model.joblib"
)

REVIEW_MODEL_FILE = (
    DATA_DIR
    / "review_risk_model.joblib"
)


# ============================================================
# DATA FILES
# ============================================================

DELIVERY_VALIDATION_FILE = (
    DATA_DIR
    / "target_delivery_validation.csv"
)

DELIVERY_TEST_FILE = (
    DATA_DIR
    / "target_delivery_test.csv"
)

REVIEW_VALIDATION_FILE = (
    DATA_DIR
    / "target_review_validation.csv"
)

REVIEW_TEST_FILE = (
    DATA_DIR
    / "target_review_test.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

DELIVERY_GLOBAL_OUTPUT = (
    DATA_DIR
    / "delivery_shap_global.csv"
)

REVIEW_GLOBAL_OUTPUT = (
    DATA_DIR
    / "review_shap_global.csv"
)

DELIVERY_VALIDATION_LOCAL_OUTPUT = (
    DATA_DIR
    / "delivery_shap_validation_local.csv"
)

REVIEW_VALIDATION_LOCAL_OUTPUT = (
    DATA_DIR
    / "review_shap_validation_local.csv"
)

DELIVERY_TEST_LOCAL_OUTPUT = (
    DATA_DIR
    / "delivery_shap_test_local.csv"
)

REVIEW_TEST_LOCAL_OUTPUT = (
    DATA_DIR
    / "review_shap_test_local.csv"
)


# ============================================================
# CORE COLUMNS
# ============================================================

SELLER_COLUMN = "seller_id"

DATE_COLUMN = "snapshot_date"


# ============================================================
# SHAP SETTINGS
# ============================================================

TOP_GLOBAL_FEATURES = 15

TOP_LOCAL_FEATURES = 10

LOCAL_SELLERS = 5


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    """Print a formatted section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_file(
    path,
    label,
):
    """Ensure a required file exists."""

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n"
            f"{path}"
        )


def load_model_package(
    path,
    label,
    expected_target,
):
    """Load and validate a saved model package."""

    package = joblib.load(
        path
    )

    required_fields = [
        "model",
        "features",
        "target",
        "threshold",
    ]

    missing = [
        field
        for field
        in required_fields
        if field not in package
    ]

    if missing:

        raise ValueError(
            f"{label} model package missing:\n"
            +
            "\n".join(
                missing
            )
        )

    if (
        package[
            "target"
        ]
        !=
        expected_target
    ):

        raise ValueError(
            f"{label} model target does not match "
            f"{expected_target}."
        )

    print(
        f"{label} model package -> PASS"
    )

    print(
        f"{label} predictors: "
        f"{len(package['features'])}"
    )

    print(
        f"{label} threshold: "
        f"{package['threshold']:.2f}"
    )

    return package


def load_dataset(
    path,
    label,
    target,
):
    """
    Load and validate a temporal feature dataset.

    IMPORTANT:
    The prediction grain is:

        seller_id + snapshot_date

    Sellers may therefore appear multiple times across
    snapshots. Only duplicate seller/snapshot combinations
    represent invalid duplicate observations.
    """

    df = pd.read_csv(
        path
    )

    print(
        f"{label} rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        SELLER_COLUMN,
        DATE_COLUMN,
        target,
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{label} missing required columns:\n"
            +
            "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Date preparation
    # --------------------------------------------------------

    df[
        DATE_COLUMN
    ] = pd.to_datetime(
        df[
            DATE_COLUMN
        ],
        errors="coerce",
    )

    invalid_dates = (
        df[
            DATE_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        f"{label} invalid snapshot dates: "
        f"{invalid_dates}"
    )

    if invalid_dates > 0:

        raise ValueError(
            f"{label} contains invalid snapshot dates."
        )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    missing_targets = (
        df[
            target
        ]
        .isna()
        .sum()
    )

    print(
        f"{label} missing target values: "
        f"{missing_targets}"
    )

    if missing_targets > 0:

        raise ValueError(
            f"{label} contains missing target values."
        )

    invalid_targets = (
        set(
            df[
                target
            ]
            .unique()
        )
        -
        {0, 1}
    )

    print(
        f"{label} invalid target values: "
        f"{len(invalid_targets)}"
    )

    if invalid_targets:

        raise ValueError(
            f"{label} contains invalid target values: "
            f"{invalid_targets}"
        )

    # --------------------------------------------------------
    # Temporal prediction grain validation
    # --------------------------------------------------------
    #
    # DO NOT check seller_id alone.
    #
    # The same seller should appear at multiple dates.
    # --------------------------------------------------------

    duplicate_seller_snapshots = (
        df[
            [
                SELLER_COLUMN,
                DATE_COLUMN,
            ]
        ]
        .duplicated()
        .sum()
    )

    print(
        f"{label} duplicate seller/snapshot rows: "
        f"{duplicate_seller_snapshots}"
    )

    if duplicate_seller_snapshots > 0:

        raise ValueError(
            f"{label} contains "
            f"{duplicate_seller_snapshots} duplicate "
            "seller/snapshot observations."
        )

    print(
        f"{label} seller/snapshot grain -> PASS"
    )

    # --------------------------------------------------------
    # Informational population statistics
    # --------------------------------------------------------

    print(
        f"{label} unique sellers: "
        f"{df[SELLER_COLUMN].nunique():,}"
    )

    print(
        f"{label} unique snapshots: "
        f"{df[DATE_COLUMN].nunique():,}"
    )

    return df


def prepare_feature_matrix(
    df,
    features,
):
    """
    Return model predictors in the exact order stored
    inside the model package.
    """

    missing_features = [
        feature
        for feature
        in features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Dataset missing model features:\n"
            +
            "\n".join(
                missing_features
            )
        )

    X = (
        df[
            features
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Validate numeric representation.
    # --------------------------------------------------------

    non_numeric = (
        X
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    if non_numeric:

        raise ValueError(
            "Non-numeric model features found:\n"
            +
            "\n".join(
                non_numeric
            )
        )

    return X


def extract_positive_class_shap_values(
    explainer,
    X,
):
    """
    Obtain SHAP values for the positive class.

    SHAP versions can return binary classification results
    in different shapes.

    This function normalizes supported binary outputs into:

        observations x features
    """

    shap_values = (
        explainer.shap_values(
            X
        )
    )

    # --------------------------------------------------------
    # Older / common SHAP representation:
    #
    # list[class_0_values, class_1_values]
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list,
    ):

        if len(
            shap_values
        ) != 2:

            raise ValueError(
                "Expected two SHAP class arrays "
                "for binary classification."
            )

        return np.asarray(
            shap_values[1]
        )

    shap_array = np.asarray(
        shap_values
    )

    # --------------------------------------------------------
    # Already observations x features
    # --------------------------------------------------------

    if shap_array.ndim == 2:

        return shap_array

    # --------------------------------------------------------
    # Three-dimensional representations
    # --------------------------------------------------------

    if shap_array.ndim == 3:

        # observations x features x classes
        if (
            shap_array.shape[2]
            ==
            2
        ):

            return shap_array[
                :,
                :,
                1,
            ]

        # classes x observations x features
        if (
            shap_array.shape[0]
            ==
            2
        ):

            return shap_array[
                1,
                :,
                :,
            ]

    raise ValueError(
        f"Unexpected SHAP output shape: "
        f"{shap_array.shape}"
    )


def build_global_shap_table(
    shap_values,
    X,
    features,
):
    """
    Build global SHAP importance.

    mean_absolute_shap:
        average magnitude of feature contribution.

    mean_signed_shap:
        average direction of contribution.
    """

    mean_absolute_shap = (
        np.mean(
            np.abs(
                shap_values
            ),
            axis=0,
        )
    )

    mean_signed_shap = (
        np.mean(
            shap_values,
            axis=0,
        )
    )

    feature_mean = (
        X[
            features
        ]
        .mean()
        .to_numpy()
    )

    table = pd.DataFrame(
        {
            "feature":
                features,

            "mean_absolute_shap":
                mean_absolute_shap,

            "mean_signed_shap":
                mean_signed_shap,

            "mean_feature_value":
                feature_mean,
        }
    )

    table = (
        table
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    table[
        "global_rank"
    ] = (
        table.index
        + 1
    )

    return table


def choose_local_rows(
    df,
    probabilities,
    number_of_sellers,
):
    """
    Select deterministic validation/test examples.

    We select observations from:

        lowest predicted risk
        lower-middle risk
        middle risk
        upper-middle risk
        highest predicted risk

    This gives us a range of local explanations.

    IMPORTANT:
    These are observations, not necessarily unique sellers,
    because the underlying temporal grain is seller/snapshot.
    """

    working = df[
        [
            SELLER_COLUMN,
            DATE_COLUMN,
        ]
    ].copy()

    working[
        "risk_probability"
    ] = probabilities

    working = (
        working
        .sort_values(
            "risk_probability"
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(working)
        <=
        number_of_sellers
    ):

        return working

    # --------------------------------------------------------
    # Select evenly distributed positions.
    # --------------------------------------------------------

    positions = [
        0,

        len(working)
        // 4,

        len(working)
        // 2,

        (
            3
            *
            len(working)
            //
            4
        ),

        len(working)
        -
        1,
    ]

    positions = sorted(
        set(
            positions
        )
    )

    positions = positions[
        :number_of_sellers
    ]

    return (
        working
        .iloc[
            positions
        ]
        .copy()
    )


def build_local_shap_table(
    df,
    shap_values,
    probabilities,
    features,
):
    """
    Build local seller/snapshot SHAP explanations.

    One row represents:

        seller
        snapshot
        feature
        feature contribution
    """

    result_rows = []

    for row_position in range(
        len(df)
    ):

        seller_id = df.iloc[
            row_position
        ][
            SELLER_COLUMN
        ]

        snapshot_date = df.iloc[
            row_position
        ][
            DATE_COLUMN
        ]

        probability = probabilities[
            row_position
        ]

        feature_values = (
            df.iloc[
                row_position
            ][
                features
            ]
        )

        row_shap = (
            shap_values[
                row_position
            ]
        )

        # ----------------------------------------------------
        # Rank features by absolute contribution.
        # ----------------------------------------------------

        feature_order = np.argsort(
            np.abs(
                row_shap
            )
        )[
            ::-1
        ]

        feature_order = feature_order[
            :TOP_LOCAL_FEATURES
        ]

        for feature_index in feature_order:

            feature = features[
                feature_index
            ]

            shap_value = row_shap[
                feature_index
            ]

            if shap_value > 0:

                direction = (
                    "INCREASES_RISK"
                )

            elif shap_value < 0:

                direction = (
                    "DECREASES_RISK"
                )

            else:

                direction = (
                    "NO_DIRECTIONAL_EFFECT"
                )

            result_rows.append(
                {
                    SELLER_COLUMN:
                        seller_id,

                    DATE_COLUMN:
                        snapshot_date,

                    "risk_probability":
                        probability,

                    "feature":
                        feature,

                    "feature_value":
                        feature_values[
                            feature
                        ],

                    "shap_value":
                        shap_value,

                    "direction":
                        direction,
                }
            )

    return pd.DataFrame(
        result_rows
    )


def explain_model(
    label,
    model_package,
    validation_df,
    test_df,
    global_output,
    validation_local_output,
    test_local_output,
):
    """
    Run the SHAP workflow for one target-specific model.
    """

    features = (
        model_package[
            "features"
        ]
    )

    pipeline = (
        model_package[
            "model"
        ]
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    print_section(
        f"{label.upper()} FEATURE MATRICES"
    )

    X_validation = (
        prepare_feature_matrix(
            validation_df,
            features,
        )
    )

    X_test = (
        prepare_feature_matrix(
            test_df,
            features,
        )
    )

    print(
        f"{label} validation matrix: "
        f"{X_validation.shape}"
    )

    print(
        f"{label} test matrix: "
        f"{X_test.shape}"
    )

    # ========================================================
    # EXTRACT PIPELINE COMPONENTS
    # ========================================================

    print_section(
        f"{label.upper()} MODEL COMPONENTS"
    )

    if not hasattr(
        pipeline,
        "named_steps",
    ):

        raise ValueError(
            f"{label} saved model is not a pipeline "
            "with named steps."
        )

    if "imputer" not in (
        pipeline.named_steps
    ):

        raise ValueError(
            f"{label} pipeline does not contain "
            "an imputer step."
        )

    if "model" not in (
        pipeline.named_steps
    ):

        raise ValueError(
            f"{label} pipeline does not contain "
            "a model step."
        )

    imputer = (
        pipeline.named_steps[
            "imputer"
        ]
    )

    forest = (
        pipeline.named_steps[
            "model"
        ]
    )

    print(
        "Imputer component -> FOUND"
    )

    print(
        "Random Forest component -> FOUND"
    )

    # ========================================================
    # APPLY SAVED TRAINING IMPUTER
    # ========================================================
    #
    # We do NOT fit the imputer again.
    #
    # transform() applies the already-learned training
    # imputation parameters.
    # ========================================================

    print_section(
        f"{label.upper()} APPLYING SAVED IMPUTATION"
    )

    X_validation_imputed = (
        imputer.transform(
            X_validation
        )
    )

    X_test_imputed = (
        imputer.transform(
            X_test
        )
    )

    X_validation_model = (
        pd.DataFrame(
            X_validation_imputed,
            columns=features,
            index=X_validation.index,
        )
    )

    X_test_model = (
        pd.DataFrame(
            X_test_imputed,
            columns=features,
            index=X_test.index,
        )
    )

    print(
        "Validation transformation -> PASS"
    )

    print(
        "Test transformation -> PASS"
    )

    # ========================================================
    # CREATE TREE EXPLAINER
    # ========================================================

    print_section(
        f"{label.upper()} TREE SHAP EXPLAINER"
    )

    explainer = (
        shap.TreeExplainer(
            forest
        )
    )

    print(
        "TreeExplainer created -> PASS"
    )

    # ========================================================
    # VALIDATION SHAP
    # ========================================================

    print_section(
        f"{label.upper()} VALIDATION SHAP"
    )

    validation_shap = (
        extract_positive_class_shap_values(
            explainer,
            X_validation_model,
        )
    )

    print(
        f"Validation SHAP matrix: "
        f"{validation_shap.shape}"
    )

    expected_shape = (
        len(
            X_validation_model
        ),
        len(
            features
        ),
    )

    if validation_shap.shape != expected_shape:

        raise ValueError(
            f"Unexpected validation SHAP shape. "
            f"Expected {expected_shape}, "
            f"got {validation_shap.shape}."
        )

    print(
        "Validation SHAP dimensions -> PASS"
    )

    # ========================================================
    # GLOBAL VALIDATION IMPORTANCE
    # ========================================================

    print_section(
        f"{label.upper()} GLOBAL SHAP IMPORTANCE"
    )

    global_table = (
        build_global_shap_table(
            validation_shap,
            X_validation_model,
            features,
        )
    )

    print(
        global_table
        .head(
            TOP_GLOBAL_FEATURES
        )
        .round(6)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # VALIDATION PROBABILITIES
    # ========================================================

    validation_probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    # ========================================================
    # SELECT VALIDATION OBSERVATIONS
    # ========================================================

    print_section(
        f"{label.upper()} SELECTING LOCAL VALIDATION OBSERVATIONS"
    )

    validation_selection = (
        choose_local_rows(
            validation_df,
            validation_probabilities,
            LOCAL_SELLERS,
        )
    )

    print(
        validation_selection
        .round(6)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Important:
    #
    # choose_local_rows() reset its own index.
    # We therefore recover the original row positions by
    # matching seller + snapshot + probability.
    # --------------------------------------------------------

    validation_positions = []

    for _, selected_row in (
        validation_selection.iterrows()
    ):

        matches = (
            (
                validation_df[
                    SELLER_COLUMN
                ]
                ==
                selected_row[
                    SELLER_COLUMN
                ]
            )
            &
            (
                validation_df[
                    DATE_COLUMN
                ]
                ==
                selected_row[
                    DATE_COLUMN
                ]
            )
            &
            (
                np.isclose(
                    validation_probabilities,
                    selected_row[
                        "risk_probability"
                    ],
                )
            )
        )

        matching_positions = (
            np.where(
                matches
            )[0]
        )

        if len(
            matching_positions
        ) == 0:

            raise ValueError(
                "Could not recover local validation observation."
            )

        validation_positions.append(
            matching_positions[0]
        )

    local_validation_df = (
        validation_df
        .iloc[
            validation_positions
        ]
        .copy()
    )

    local_validation_shap = (
        validation_shap[
            validation_positions
        ]
    )

    local_validation_probabilities = (
        validation_probabilities[
            validation_positions
        ]
    )

    validation_local_table = (
        build_local_shap_table(
            local_validation_df,
            local_validation_shap,
            local_validation_probabilities,
            features,
        )
    )

    print_section(
        f"{label.upper()} LOCAL VALIDATION EXPLANATIONS"
    )

    print(
        validation_local_table
        .round(6)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # TEST SHAP
    # ========================================================

    print_section(
        f"{label.upper()} OUT-OF-TIME TEST SHAP"
    )

    test_shap = (
        extract_positive_class_shap_values(
            explainer,
            X_test_model,
        )
    )

    print(
        f"Test SHAP matrix: "
        f"{test_shap.shape}"
    )

    expected_test_shape = (
        len(
            X_test_model
        ),
        len(
            features
        ),
    )

    if test_shap.shape != expected_test_shape:

        raise ValueError(
            f"Unexpected test SHAP shape. "
            f"Expected {expected_test_shape}, "
            f"got {test_shap.shape}."
        )

    print(
        "Test SHAP dimensions -> PASS"
    )

    test_probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    # ========================================================
    # SELECT TEST OBSERVATIONS
    # ========================================================

    print_section(
        f"{label.upper()} SELECTING LOCAL TEST OBSERVATIONS"
    )

    test_selection = (
        choose_local_rows(
            test_df,
            test_probabilities,
            LOCAL_SELLERS,
        )
    )

    print(
        test_selection
        .round(6)
        .to_string(
            index=False
        )
    )

    test_positions = []

    for _, selected_row in (
        test_selection.iterrows()
    ):

        matches = (
            (
                test_df[
                    SELLER_COLUMN
                ]
                ==
                selected_row[
                    SELLER_COLUMN
                ]
            )
            &
            (
                test_df[
                    DATE_COLUMN
                ]
                ==
                selected_row[
                    DATE_COLUMN
                ]
            )
            &
            (
                np.isclose(
                    test_probabilities,
                    selected_row[
                        "risk_probability"
                    ],
                )
            )
        )

        matching_positions = (
            np.where(
                matches
            )[0]
        )

        if len(
            matching_positions
        ) == 0:

            raise ValueError(
                "Could not recover local test observation."
            )

        test_positions.append(
            matching_positions[0]
        )

    local_test_df = (
        test_df
        .iloc[
            test_positions
        ]
        .copy()
    )

    local_test_shap = (
        test_shap[
            test_positions
        ]
    )

    local_test_probabilities = (
        test_probabilities[
            test_positions
        ]
    )

    test_local_table = (
        build_local_shap_table(
            local_test_df,
            local_test_shap,
            local_test_probabilities,
            features,
        )
    )

    print_section(
        f"{label.upper()} LOCAL TEST EXPLANATIONS"
    )

    print(
        test_local_table
        .round(6)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    print_section(
        f"SAVING {label.upper()} SHAP RESULTS"
    )

    global_table.to_csv(
        global_output,
        index=False,
    )

    validation_local_table.to_csv(
        validation_local_output,
        index=False,
    )

    test_local_table.to_csv(
        test_local_output,
        index=False,
    )

    print(
        f"Global SHAP saved to:\n"
        f"{global_output}"
    )

    print(
        f"\nValidation local SHAP saved to:\n"
        f"{validation_local_output}"
    )

    print(
        f"\nTest local SHAP saved to:\n"
        f"{test_local_output}"
    )

    return {
        "global":
            global_table,

        "validation_local":
            validation_local_table,

        "test_local":
            test_local_table,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TARGET-SPECIFIC SHAP EXPLAINABILITY ANALYSIS"
    )
    print("=" * 70)

    # ========================================================
    # 1. CHECK REQUIRED FILES
    # ========================================================

    print_section(
        "CHECKING REQUIRED INPUTS"
    )

    required_files = [
        (
            DELIVERY_MODEL_FILE,
            "Delivery model",
        ),
        (
            REVIEW_MODEL_FILE,
            "Review model",
        ),
        (
            DELIVERY_VALIDATION_FILE,
            "Delivery validation data",
        ),
        (
            DELIVERY_TEST_FILE,
            "Delivery test data",
        ),
        (
            REVIEW_VALIDATION_FILE,
            "Review validation data",
        ),
        (
            REVIEW_TEST_FILE,
            "Review test data",
        ),
    ]

    for path, label in required_files:

        validate_file(
            path,
            label,
        )

        print(
            f"{label} -> FOUND"
        )

    # ========================================================
    # 2. LOAD MODELS
    # ========================================================

    print_section(
        "LOADING TARGET-SPECIFIC MODELS"
    )

    delivery_package = (
        load_model_package(
            DELIVERY_MODEL_FILE,
            "Delivery",
            "future_delivery_risk",
        )
    )

    review_package = (
        load_model_package(
            REVIEW_MODEL_FILE,
            "Review",
            "future_review_risk",
        )
    )

    # ========================================================
    # 3. LOAD DATA
    # ========================================================

    print_section(
        "LOADING EXPLANATION DATA"
    )

    delivery_validation = (
        load_dataset(
            DELIVERY_VALIDATION_FILE,
            "Delivery validation",
            "future_delivery_risk",
        )
    )

    delivery_test = (
        load_dataset(
            DELIVERY_TEST_FILE,
            "Delivery test",
            "future_delivery_risk",
        )
    )

    review_validation = (
        load_dataset(
            REVIEW_VALIDATION_FILE,
            "Review validation",
            "future_review_risk",
        )
    )

    review_test = (
        load_dataset(
            REVIEW_TEST_FILE,
            "Review test",
            "future_review_risk",
        )
    )

    # ========================================================
    # 4. DATASET GRAIN SUMMARY
    # ========================================================

    print_section(
        "TEMPORAL DATA GRAIN SUMMARY"
    )

    datasets = [
        (
            "Delivery validation",
            delivery_validation,
        ),
        (
            "Delivery test",
            delivery_test,
        ),
        (
            "Review validation",
            review_validation,
        ),
        (
            "Review test",
            review_test,
        ),
    ]

    for label, df in datasets:

        print(
            f"{label}: "
            f"{len(df):,} observations, "
            f"{df[SELLER_COLUMN].nunique():,} sellers, "
            f"{df[DATE_COLUMN].nunique():,} snapshots"
        )

    print(
        "\nTemporal grain:"
    )

    print(
        "seller_id + snapshot_date"
    )

    print(
        "Repeated sellers across different snapshots are valid."
    )

    # ========================================================
    # 5. TEST PERIOD PROTECTION
    # ========================================================

    print_section(
        "OUT-OF-TIME TEST PROTECTION"
    )

    delivery_test_start = (
        delivery_test[
            DATE_COLUMN
        ]
        .min()
    )

    delivery_test_end = (
        delivery_test[
            DATE_COLUMN
        ]
        .max()
    )

    review_test_start = (
        review_test[
            DATE_COLUMN
        ]
        .min()
    )

    review_test_end = (
        review_test[
            DATE_COLUMN
        ]
        .max()
    )

    print(
        f"Delivery test: "
        f"{delivery_test_start.date()} "
        f"-> "
        f"{delivery_test_end.date()}"
    )

    print(
        f"Review test: "
        f"{review_test_start.date()} "
        f"-> "
        f"{review_test_end.date()}"
    )

    print(
        "Test explanations are diagnostic only."
    )

    # ========================================================
    # 6. DELIVERY SHAP
    # ========================================================

    delivery_results = explain_model(
        "Delivery",
        delivery_package,
        delivery_validation,
        delivery_test,
        DELIVERY_GLOBAL_OUTPUT,
        DELIVERY_VALIDATION_LOCAL_OUTPUT,
        DELIVERY_TEST_LOCAL_OUTPUT,
    )

    # ========================================================
    # 7. REVIEW SHAP
    # ========================================================

    review_results = explain_model(
        "Review",
        review_package,
        review_validation,
        review_test,
        REVIEW_GLOBAL_OUTPUT,
        REVIEW_VALIDATION_LOCAL_OUTPUT,
        REVIEW_TEST_LOCAL_OUTPUT,
    )

    # ========================================================
    # 8. GLOBAL COMPARISON
    # ========================================================

    print_section(
        "TARGET-SPECIFIC GLOBAL SHAP COMPARISON"
    )

    delivery_global = (
        delivery_results[
            "global"
        ]
        .head(
            TOP_GLOBAL_FEATURES
        )
        [
            [
                "feature",
                "mean_absolute_shap",
                "mean_signed_shap",
            ]
        ]
        .copy()
    )

    review_global = (
        review_results[
            "global"
        ]
        .head(
            TOP_GLOBAL_FEATURES
        )
        [
            [
                "feature",
                "mean_absolute_shap",
                "mean_signed_shap",
            ]
        ]
        .copy()
    )

    delivery_global[
        "model"
    ] = "DELIVERY"

    review_global[
        "model"
    ] = "REVIEW"

    combined = pd.concat(
        [
            delivery_global,
            review_global,
        ],
        ignore_index=True,
    )

    print(
        combined
        .round(6)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 9. INTERPRETATION GUIDE
    # ========================================================

    print_section(
        "SHAP INTERPRETATION GUIDE"
    )

    print(
        "Mean absolute SHAP:"
    )

    print(
        "Average magnitude of a feature's model contribution."
    )

    print(
        "\nMean signed SHAP:"
    )

    print(
        "Average direction of the feature's contribution "
        "toward or away from the positive-risk class."
    )

    print(
        "\nPositive SHAP:"
    )

    print(
        "The feature pushes that prediction toward greater risk."
    )

    print(
        "\nNegative SHAP:"
    )

    print(
        "The feature pushes that prediction toward lower risk."
    )

    print(
        "\nImportant:"
    )

    print(
        "SHAP explains model behavior."
    )

    print(
        "It does not establish that a feature causes the outcome."
    )

    print(
        "A feature can be highly influential because the model "
        "has learned an association rather than a causal relationship."
    )

    # ========================================================
    # 10. FINAL
    # ========================================================

    print_section(
        "TARGET-SPECIFIC SHAP ANALYSIS COMPLETE"
    )

    print(
        "Delivery global explanations -> SAVED"
    )

    print(
        "Review global explanations -> SAVED"
    )

    print(
        "Delivery validation local explanations -> SAVED"
    )

    print(
        "Review validation local explanations -> SAVED"
    )

    print(
        "Delivery test local explanations -> SAVED"
    )

    print(
        "Review test local explanations -> SAVED"
    )

    print(
        "\nNo model was retrained."
    )

    print(
        "No threshold was changed."
    )

    print(
        "No SHAP result was used to tune either model."
    )

    print(
        "The final test period remains diagnostic-only."
    )

    print(
        "The next stage is baseline-model comparison."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()