from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_seller_features():

    file_path = (
        PROCESSED_DATA_DIR
        / "seller_features.csv"
    )

    return pd.read_csv(file_path)


# =========================================================
# CATALOG RISK
# =========================================================

def calculate_catalog_risk(df):

    df = df.copy()

    df["catalog_risk"] = (
        100
        -
        df["catalog_completeness"]
    )

    return df


# =========================================================
# COMMERCIAL EXPOSURE
# =========================================================

def calculate_commercial_exposure(df):

    df = df.copy()

    revenue_95th = (
        df["total_revenue"]
        .quantile(0.95)
    )

    df["commercial_exposure"] = (
        df["total_revenue"]
        /
        revenue_95th
        * 100
    )

    df["commercial_exposure"] = (
        df["commercial_exposure"]
        .clip(upper=100)
    )

    return df


# =========================================================
# ORDER EXPOSURE
# =========================================================

def calculate_order_exposure(df):

    df = df.copy()

    orders_95th = (
        df["total_orders"]
        .quantile(0.95)
    )

    df["order_exposure"] = (
        df["total_orders"]
        /
        orders_95th
        * 100
    )

    df["order_exposure"] = (
        df["order_exposure"]
        .clip(upper=100)
    )

    return df


# =========================================================
# SELLER RISK SCORE
# =========================================================

def calculate_risk_score(df):

    df = df.copy()

    df["seller_risk_score"] = (
        (df["catalog_risk"] * 0.50)
        +
        (df["commercial_exposure"] * 0.30)
        +
        (df["order_exposure"] * 0.20)
    )

    return df


# =========================================================
# RISK CATEGORY
# =========================================================

def assign_risk_category(df):

    df = df.copy()

    def classify(score):

        if score >= 70:
            return "HIGH"

        elif score >= 40:
            return "MEDIUM"

        else:
            return "LOW"

    df["risk_category"] = (
        df["seller_risk_score"]
        .apply(classify)
    )

    return df


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(df):

    output_file = (
        PROCESSED_DATA_DIR
        / "seller_risk_scores.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nRisk score dataset saved to:\n"
        f"{output_file}"
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("=" * 70)
    print("SELLER RISK SCORING")
    print("=" * 70)

    df = load_seller_features()

    df = calculate_catalog_risk(df)

    df = calculate_commercial_exposure(df)

    df = calculate_order_exposure(df)

    df = calculate_risk_score(df)

    df = assign_risk_category(df)

    # -----------------------------------------------------
    # DISPLAY TOP SELLERS BY RISK
    # -----------------------------------------------------

    top_risk = (
        df[
            [
                "seller_id",
                "catalog_completeness",
                "catalog_risk",
                "commercial_exposure",
                "order_exposure",
                "seller_risk_score",
                "risk_category"
            ]
        ]
        .sort_values(
            "seller_risk_score",
            ascending=False
        )
        .head(20)
    )

    print("\nTop 20 sellers by risk score:")

    print(
        top_risk
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # RISK DISTRIBUTION
    # -----------------------------------------------------

    print("\nRisk category distribution:")

    print(
        df["risk_category"]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    save_results(df)

    print("\n" + "=" * 70)
    print("SELLER RISK SCORING COMPLETE")
    print("=" * 70)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()