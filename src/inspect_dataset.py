
from pathlib import Path
import pandas as pd

# Locate the project root directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Exact dataset path based on your VS Code folder structure
DATASET_PATH = PROJECT_DIR / "data" / "archive" / "spam_assassin.csv"


def inspect_dataset():
    """Load and inspect the SpamAssassin dataset."""

    print("=" * 60)
    print("SPAMSHIELD ML - DATASET INSPECTION")
    print("=" * 60)

    # Check whether the dataset exists
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Please check the CSV filename and folder location."
        )

    print(f"\nDataset path: {DATASET_PATH}")

    # Load the CSV file
    df = pd.read_csv(DATASET_PATH)

    # 1. Dataset dimensions
    print("\n1. DATASET DIMENSIONS")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # 2. Column names
    print("\n2. COLUMN NAMES")
    for column in df.columns:
        print(f"- {column}")

    # 3. First five records
    print("\n3. FIRST FIVE ROWS")
    print(df.head().to_string(index=False))

    # 4. Data types
    print("\n4. DATA TYPES")
    print(df.dtypes)

    # 5. Missing values
    print("\n5. MISSING VALUES")
    print(df.isnull().sum())

    # 6. Exact duplicate rows
    print("\n6. EXACT DUPLICATE ROWS")
    print(df.duplicated().sum())

    # 7. Unique values in each column
    print("\n7. UNIQUE VALUE COUNTS")
    for column in df.columns:
        print(f"\n{column}:")
        print(f"Unique values: {df[column].nunique(dropna=True)}")

    # 8. Preview potential label columns
    print("\n8. POSSIBLE LABEL DISTRIBUTIONS")
    for column in df.columns:
        name = str(column).lower()
        if any(word in name for word in ["label", "target", "class", "spam", "category"]):
            print(f"\nColumn: {column}")
            print(df[column].value_counts(dropna=False).head(20))

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
