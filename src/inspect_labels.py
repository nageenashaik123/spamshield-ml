
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "data" / "archive" / "spam_assassin.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("SPAMSHIELD ML - LABEL MAPPING INSPECTION")
print("=" * 70)

for label in sorted(df["target"].unique()):
    group = df[df["target"] == label]

    print(f"\n{'=' * 25} TARGET {label} {'=' * 25}")
    print(f"Number of emails: {len(group)}")

    # Display three examples from each class
    for i, (_, row) in enumerate(group.head(3).iterrows(), start=1):
        email_text = str(row["text"])

        print(f"\nExample {i}")
        print("-" * 50)

        # Display the email headers and opening text
        print(email_text[:1200])
        print("\n[End of preview]")

print("\nInspection completed.")
