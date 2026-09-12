from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# =========================================================
# FIND DATASETS
# =========================================================

def get_dataset_files():
    """
    Find all CSV datasets in the raw data directory.
    """

    return sorted(RAW_DATA_DIR.glob("*.csv"))


# =========================================================
# INSPECT DATASET
# =========================================================

def inspect_dataset(file_path):
    """
    Inspect the structure and basic characteristics
    of a single dataset.
    """

    df = pd.read_csv(file_path)

    print("=" * 70)
    print(f"DATASET: {file_path.name}")
    print("=" * 70)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nUnique values:")
    print(df.nunique())

    print("\nMissing values by column:")
    print(df.isna().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\n" + "-" * 70)


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    dataset_files = get_dataset_files()

    print("=" * 70)
    print("OLIST DATA DICTIONARY INSPECTION")
    print("=" * 70)

    print(f"\nDatasets found: {len(dataset_files)}")

    for file_path in dataset_files:
        inspect_dataset(file_path)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()