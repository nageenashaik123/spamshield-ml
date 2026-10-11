
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_DIR / "data" / "archive" / "spam_assassin.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "spam_assassin_clean.csv"


def clean_dataset():
    print("=" * 60)
    print("SPAMSHIELD ML - DATA CLEANING")
    print("=" * 60)

    # Load the original dataset
    df = pd.read_csv(RAW_PATH)

    # Verify the expected columns
    required_columns = {"text", "target"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    print(f"\nOriginal rows: {len(df)}")

    # Check for missing values
    print("\nMissing values:")
    print(df[["text", "target"]].isnull().sum())

    if df[["text", "target"]].isnull().any().any():
        raise ValueError(
            "Missing text or target values found. "
            "Review these rows before cleaning."
        )

    # Check labels
    valid_labels = {0, 1}
    actual_labels = set(df["target"].unique())

    if actual_labels != valid_labels:
        raise ValueError(f"Unexpected target labels: {actual_labels}")

    # Verify identical email text never has conflicting labels
    labels_per_text = df.groupby("text")["target"].nunique()

    if (labels_per_text > 1).any():
        raise ValueError(
            "Some identical email texts have conflicting labels. "
            "Review them before removing duplicates."
        )

    # Remove exact duplicate rows only
    before = len(df)
    df = df.drop_duplicates(subset=["text", "target"]).copy()
    removed = before - len(df)

    # Keep only the required columns
    df = df[["text", "target"]].reset_index(drop=True)

    # Create output directory and save the cleaned data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDuplicate rows removed: {removed}")
    print(f"Cleaned rows: {len(df)}")
    print(f"Unique email texts: {df['text'].nunique()}")

    print("\nClass distribution:")
    print(df["target"].value_counts().sort_index())

    print("\nClass distribution (%):")
    print(
        (df["target"].value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )

    print(f"\nSaved cleaned dataset to:\n{OUTPUT_PATH}")
    print("\nCleaning completed successfully!")


if __name__ == "__main__":
    clean_dataset()
