from pathlib import Path
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# =========================================================
# DATASET INVENTORY
# =========================================================

def get_dataset_files():
    """
    Find all CSV datasets inside the raw data directory.
    """

    return sorted(RAW_DATA_DIR.glob("*.csv"))


# =========================================================
# DATASET INSPECTION
# =========================================================

def inspect_dataset(file_path):
    """
    Load a dataset and return basic structural information.
    """

    df = pd.read_csv(file_path)

    information = {
        "dataset": file_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isna().sum().sum(),
        "duplicate_rows": df.duplicated().sum()
    }

    return information


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    dataset_files = get_dataset_files()

    print("=" * 70)
    print("OLIST DATASET INVENTORY")
    print("=" * 70)

    print(f"\nRaw data directory:")
    print(RAW_DATA_DIR)

    print(f"\nDatasets found: {len(dataset_files)}")

    print("\n" + "-" * 70)

    inventory = []

    for file_path in dataset_files:

        print(f"\nInspecting: {file_path.name}")

        information = inspect_dataset(file_path)

        inventory.append(information)

        print(f"Rows: {information['rows']:,}")
        print(f"Columns: {information['columns']}")
        print(f"Missing values: {information['missing_values']:,}")
        print(f"Duplicate rows: {information['duplicate_rows']:,}")

    # Convert results into a DataFrame
    inventory_df = pd.DataFrame(inventory)

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(inventory_df.to_string(index=False))


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()