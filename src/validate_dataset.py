
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "data" / "archive" / "spam_assassin.csv"


def validate_dataset():
    df = pd.read_csv(DATASET_PATH)

    print("=" * 60)
    print("SPAMSHIELD ML - DATASET VALIDATION")
    print("=" * 60)

    # Check required columns
    required_columns = {"text", "target"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_columns}, "
            f"but found {set(df.columns)}"
        )

    # Check labels
    print("\n1. LABEL VALUES")
    print(df["target"].value_counts().sort_index())

    print("\n2. LABEL DISTRIBUTION (%)")
    print(
        (df["target"].value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )

    # Confirm label consistency for identical email text
    print("\n3. IDENTICAL TEXT WITH DIFFERENT LABELS")
    labels_per_text = df.groupby("text")["target"].nunique()
    conflicting_texts = labels_per_text[labels_per_text > 1]

    print("Number of conflicting email texts:", len(conflicting_texts))

    # Inspect exact duplicate rows
    print("\n4. EXACT DUPLICATE ROWS")
    print("Duplicate rows:", df.duplicated().sum())

    # Count repeated email texts
    print("\n5. REPEATED EMAIL TEXTS")
    print("Rows with repeated text:", df["text"].duplicated().sum())
    print("Unique email texts:", df["text"].nunique())

    # Show a few examples for manual review
    if len(conflicting_texts) > 0:
        print("\n6. EXAMPLES OF CONFLICTING LABELS")

        for text in conflicting_texts.index[:3]:
            print("\nText preview:", repr(text[:250]))
            print(
                "Labels:",
                df.loc[df["text"] == text, "target"].unique()
            )

    print("\n7. EXAMPLE EMAIL LENGTHS")
    print(df["text"].str.len().describe())

    print("\nValidation completed.")


if __name__ == "__main__":
    validate_dataset()
