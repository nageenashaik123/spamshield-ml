
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "spam_assassin_clean.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Validate required columns
if not {"text", "target"}.issubset(df.columns):
    raise ValueError("Dataset must contain 'text' and 'target' columns.")

if df["text"].isna().any() or df["target"].isna().any():
    raise ValueError("Missing values found. Clean the dataset first.")

# Separate input text and target labels
X = df["text"].astype(str)
y = df["target"].astype(int)

# First, reserve 20% for the final test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Split the remaining 80% into training and validation sets
# Final proportions: 60% training, 20% validation, 20% testing
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

# Save splits
splits = {
    "train": (X_train, y_train),
    "validation": (X_val, y_val),
    "test": (X_test, y_test),
}

for name, (X_part, y_part) in splits.items():
    split_df = pd.DataFrame(
        {
            "text": X_part.to_numpy(),
            "target": y_part.to_numpy(),
        }
    )

    output_path = OUTPUT_DIR / f"{name}.csv"
    split_df.to_csv(output_path, index=False)

    print(f"\n{name.upper()} SET")
    print(f"Emails: {len(split_df)}")
    print("Class counts:")
    print(split_df["target"].value_counts().sort_index())
    print("Saved to:", output_path)

print("\nData splitting completed successfully!")
