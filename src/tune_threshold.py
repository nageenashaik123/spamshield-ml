
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# =====================================================
# 1. PATHS AND DATA
# =====================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train["text"].fillna("").astype(str)
y_train = train["target"].astype(int)

X_val = val["text"].fillna("").astype(str)
y_val = val["target"].astype(int)

print("=" * 65)
print("SPAMSHIELD ML - THRESHOLD TUNING")
print("=" * 65)
print("Training emails:", len(X_train))
print("Validation emails:", len(X_val))


# =====================================================
# 2. WORD TF-IDF
# =====================================================

word_vec = TfidfVectorizer(
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

X_train_word = word_vec.fit_transform(X_train)
X_val_word = word_vec.transform(X_val)


# =====================================================
# 3. CHARACTER TF-IDF
# =====================================================

char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_char = char_vec.fit_transform(X_train)
X_val_char = char_vec.transform(X_val)


# =====================================================
# 4. COMBINE FEATURES
# =====================================================

X_train_all = hstack(
    [X_train_word, X_train_char],
    format="csr",
    dtype=np.float32,
)

X_val_all = hstack(
    [X_val_word, X_val_char],
    format="csr",
    dtype=np.float32,
)

print("\nTraining feature matrix:", X_train_all.shape)
print("Validation feature matrix:", X_val_all.shape)


# =====================================================
# 5. TRAIN SVM
# =====================================================

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
)

print("\nTraining Linear SVM...")
model.fit(X_train_all, y_train)

scores = model.decision_function(X_val_all)


# =====================================================
# 6. COMPARE THRESHOLDS
# =====================================================

# A prediction is Spam when score >= threshold.
# Larger thresholds generally reduce false positives,
# but may cause more spam emails to be missed.

targets = [0.005, 0.01, 0.02, 0.03, 0.05]

thresholds = np.unique(
    np.concatenate([
        scores,
        [np.nextafter(scores.max(), np.inf)],
    ])
)

results = []

for target_fpr in targets:
    candidates = []

    for threshold_value in thresholds:
        predictions = (scores >= threshold_value).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        spam_recall = recall_score(
            y_val, predictions, zero_division=0
        )
        spam_precision = precision_score(
            y_val, predictions, zero_division=0
        )
        spam_f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        if fpr <= target_fpr:
            candidates.append({
                "target_fpr": target_fpr,
                "threshold": float(threshold_value),
                "fpr": fpr,
                "spam_precision": spam_precision,
                "spam_recall": spam_recall,
                "spam_f1": spam_f1,
                "false_positives": int(fp),
                "missed_spam": int(fn),
            })

    if candidates:
        best = max(
            candidates,
            key=lambda row: (
                row["spam_f1"],
                -row["fpr"],
                row["spam_precision"],
                -abs(row["threshold"]),
            ),
        )
        results.append(best)

if not results:
    raise RuntimeError("No valid threshold candidates were found.")

comparison = pd.DataFrame(results)

print("\n" + "=" * 90)
print("VALIDATION THRESHOLD COMPARISON")
print("=" * 90)

print(
    comparison[
        [
            "target_fpr",
            "threshold",
            "fpr",
            "spam_precision",
            "spam_recall",
            "spam_f1",
            "false_positives",
            "missed_spam",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# =====================================================
# 7. SELECT OPERATING POINT
# =====================================================

# Default: strict 1% maximum validation false-positive rate.
# Inspect other rows to understand the recall trade-off.

chosen = next(
    (
        row for row in results
        if np.isclose(row["target_fpr"], 0.01)
    ),
    None,
)

if chosen is None:
    chosen = min(results, key=lambda row: row["fpr"])

print("\nSELECTED THRESHOLD")
print("-" * 45)
print("Threshold:", round(chosen["threshold"], 6))
print("Validation FPR:", f"{chosen['fpr']:.2%}")
print("Spam precision:", f"{chosen['spam_precision']:.2%}")
print("Spam recall:", f"{chosen['spam_recall']:.2%}")
print("Spam F1:", f"{chosen['spam_f1']:.4f}")
print("Legitimate emails flagged:", chosen["false_positives"])
print("Spam emails missed:", chosen["missed_spam"])


# =====================================================
# 8. SAVE MODEL AND REPORT
# =====================================================

artifact = {
    "word_vectorizer": word_vec,
    "char_vectorizer": char_vec,
    "model": model,
    "threshold": chosen["threshold"],
    "model_name": "Linear SVM",
    "feature_representation": "Word + Character TF-IDF",
    "positive_label": 1,
    "class_names": {
        0: "Not Spam",
        1: "Spam",
    },
    "validation_false_positive_rate": chosen["fpr"],
    "validation_spam_recall": chosen["spam_recall"],
    "validation_spam_precision": chosen["spam_precision"],
}

model_path = MODEL_DIR / "spam_classifier_threshold_tuned.joblib"
joblib.dump(artifact, model_path)

report_path = REPORT_DIR / "threshold_comparison.csv"
comparison.to_csv(report_path, index=False)

print("\nSaved model:", model_path)
print("Saved comparison:", report_path)
print("\nThreshold tuning completed successfully.")
