
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "spam_assassin_clean.csv"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Intended mapping: verify against dataset documentation
label_names = {
    0: "Not Spam",
    1: "Spam"
}

df["class_name"] = df["target"].map(label_names)

if df["class_name"].isna().any():
    raise ValueError("Unexpected target label found.")

# Calculate email length in characters
df["email_length"] = df["text"].str.len()

sns.set_theme(style="whitegrid")

# 1. Class distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(
    data=df,
    x="class_name",
    hue="class_name",
    order=["Not Spam", "Spam"],
    palette="Set2",
    legend=False
)
ax.set_title("Email Class Distribution")
ax.set_xlabel("Email Class")
ax.set_ylabel("Number of Emails")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.savefig(REPORT_DIR / "class_distribution.png", dpi=150)
plt.show()

# 2. Email length distribution
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="email_length",
    hue="class_name",
    bins=50,
    element="step",
    stat="count",
    common_norm=False
)
plt.title("Email Length Distribution")
plt.xlabel("Email Length (characters)")
plt.ylabel("Number of Emails")
plt.tight_layout()
plt.savefig(REPORT_DIR / "email_length_distribution.png", dpi=150)
plt.show()

# 3. Box plot for email lengths
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="class_name",
    y="email_length",
    hue="class_name",
    order=["Not Spam", "Spam"],
    legend=False
)
plt.title("Email Length by Class")
plt.xlabel("Email Class")
plt.ylabel("Email Length (characters)")
plt.tight_layout()
plt.savefig(REPORT_DIR / "email_length_boxplot.png", dpi=150)
plt.show()

# 4. Summary statistics
print("=" * 55)
print("SPAMSHIELD ML - EDA SUMMARY")
print("=" * 55)

print("\nDataset shape:", df.shape)

print("\nClass counts:")
print(df["class_name"].value_counts())

print("\nClass percentages:")
print((df["class_name"].value_counts(normalize=True) * 100).round(2))

print("\nEmail length statistics by class:")
print(
    df.groupby("class_name")["email_length"]
    .describe()
    .round(2)
)

print("\nCharts saved in:", REPORT_DIR)
print("EDA completed.")
