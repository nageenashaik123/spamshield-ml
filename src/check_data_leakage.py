
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

splits = {
    "train": set(train["text"].fillna("").astype(str)),
    "validation": set(validation["text"].fillna("").astype(str)),
    "test": set(test["text"].fillna("").astype(str)),
}

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

print("=" * 55)
print("SPAMSHIELD ML - CROSS-SPLIT DUPLICATE AUDIT")
print("=" * 55)

for first, second in pairs:
    overlap = splits[first].intersection(splits[second])
    print(f"{first} vs {second}: {len(overlap)} exact matching emails")

if any(
    splits[first].intersection(splits[second])
    for first, second in pairs
):
    print("\nWARNING: Exact text overlap found.")
else:
    print("\nPASS: No exact email text overlap across splits.")

print(
    "\nNote: This checks exact text matches, not near-duplicates "
    "or sender/thread relationships."
)
