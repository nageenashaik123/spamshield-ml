
from pathlib import Path

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
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["target"].astype(int)

X_val = val_df["text"].fillna("").astype(str)
y_val = val_df["target"].astype(int)


def evaluate_model(name, train_features, val_features):
    model = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    model.fit(train_features, y_train)

    predictions = model.predict(val_features)
    scores = model.decision_function(val_features)

    result = {
        "Experiment": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions, zero_division=0),
        "Recall": recall_score(y_val, predictions, zero_division=0),
        "F1 Score": f1_score(y_val, predictions, zero_division=0),
        "PR-AUC": average_precision_score(y_val, scores),
    }

    print(f"\n{name}")
    for metric, value in result.items():
        if metric != "Experiment":
            print(f"{metric}: {value:.4f}")

    return result


# Experiment A: word features
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

X_train_word = word_vectorizer.fit_transform(X_train)
X_val_word = word_vectorizer.transform(X_val)

results = [
    evaluate_model(
        "Word TF-IDF",
        X_train_word,
        X_val_word,
    )
]

# Experiment B: character features
char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_val_char = char_vectorizer.transform(X_val)

# Combine word and character features
X_train_combined = hstack(
    [X_train_word, X_train_char],
    format="csr",
    dtype=np.float32,
)
X_val_combined = hstack(
    [X_val_word, X_val_char],
    format="csr",
    dtype=np.float32,
)

print("\nCombined feature shape:", X_train_combined.shape)

results.append(
    evaluate_model(
        "Word + Character TF-IDF",
        X_train_combined,
        X_val_combined,
    )
)

results_df = pd.DataFrame(results).sort_values(
    "F1 Score", ascending=False
)

output_path = REPORT_DIR / "tfidf_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\n" + "=" * 55)
print("TF-IDF EXPERIMENT COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False, float_format="%.4f"))
print(f"\nResults saved to: {output_path}")
