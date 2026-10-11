
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load development data and the untouched test set
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

development_df = pd.concat(
    [train_df, val_df],
    ignore_index=True,
)

X_dev = development_df["text"].fillna("").astype(str)
y_dev = development_df["target"].astype(int)

X_test = test_df["text"].fillna("").astype(str)
y_test = test_df["target"].astype(int)

print("=" * 60)
print("SPAMSHIELD ML - FINAL TEST EVALUATION")
print("=" * 60)
print(f"Development emails: {len(X_dev)}")
print(f"Test emails: {len(X_test)}")

# 2. Fit word TF-IDF on development data only
word_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    max_features=100_000,
    sublinear_tf=True,
    stop_words="english",
    dtype=np.float32,
)

X_dev_word = word_vectorizer.fit_transform(X_dev)
X_test_word = word_vectorizer.transform(X_test)

# 3. Fit character TF-IDF on development data only
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_dev_char = char_vectorizer.fit_transform(X_dev)
X_test_char = char_vectorizer.transform(X_test)

# 4. Combine the two feature representations
X_dev_combined = hstack(
    [X_dev_word, X_dev_char],
    format="csr",
    dtype=np.float32,
)

X_test_combined = hstack(
    [X_test_word, X_test_char],
    format="csr",
    dtype=np.float32,
)

print(f"Combined training features: {X_dev_combined.shape}")
print(f"Combined test features: {X_test_combined.shape}")

# 5. Train the selected classifier
model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_dev_combined, y_dev)

# 6. Evaluate on the test set ONCE
y_pred = model.predict(X_test_combined)
y_score = model.decision_function(X_test_combined)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "PR-AUC (Average Precision)": average_precision_score(
        y_test, y_score
    ),
}

print("\nFINAL TEST METRICS")
print("-" * 40)

for name, value in metrics.items():
    print(f"{name}: {value:.4f} ({value * 100:.2f}%)")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not Spam", "Spam"],
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion matrix:")
print(cm)

# 7. Save final report
report_df = pd.DataFrame(
    [{"Metric": name, "Score": value}
     for name, value in metrics.items()]
)

report_path = REPORT_DIR / "final_test_metrics.csv"
report_df.to_csv(report_path, index=False)

# 8. Save the complete inference components
artifact = {
    "word_vectorizer": word_vectorizer,
    "char_vectorizer": char_vectorizer,
    "model": model,
    "model_name": "Linear SVM",
    "feature_representation": "Word + Character TF-IDF",
    "positive_label": 1,
    "class_names": {0: "Not Spam", 1: "Spam"},
}

model_path = MODEL_DIR / "spam_classifier_final.joblib"
joblib.dump(artifact, model_path)

print(f"\nTest report saved to: {report_path}")
print(f"Final model saved to: {model_path}")
print("\nFinal evaluation completed.")
