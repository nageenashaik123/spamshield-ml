
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
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

import joblib

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train_df["text"].fillna("").astype(str)
y_train = train_df["target"].astype(int)

X_val = val_df["text"].fillna("").astype(str)
y_val = val_df["target"].astype(int)

print("=" * 60)
print("SPAMSHIELD ML - CLASSIFIER TRAINING")
print("=" * 60)

print(f"\nTraining emails: {len(X_train)}")
print(f"Validation emails: {len(X_val)}")

# --------------------------------------------------
# 3. Convert email text into TF-IDF features
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    max_features=100_000,
    stop_words="english",
    dtype=np.float32,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print(f"\nVocabulary size: {len(vectorizer.vocabulary_)}")
print("Training matrix:", X_train_tfidf.shape)
print("Validation matrix:", X_val_tfidf.shape)

# --------------------------------------------------
# 4. Define classifiers
# --------------------------------------------------

models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    "Logistic Regression": LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
    ),
}

# --------------------------------------------------
# 5. Train and evaluate on validation data
# --------------------------------------------------

results = []
trained_models = {}

for model_name, model in models.items():
    print("\n" + "-" * 60)
    print(f"Training: {model_name}")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_val_tfidf)

    # Scores for ranking metrics
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val_tfidf)[:, 1]
    else:
        y_score = model.decision_function(X_val_tfidf)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_val, y_pred, zero_division=0
        ),
        "PR-AUC (Average Precision)": average_precision_score(
            y_val, y_score
        ),
    }

    results.append(metrics)
    trained_models[model_name] = model

    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1 Score:  {metrics['F1 Score']:.4f}")
    print(
        "PR-AUC:    "
        f"{metrics['PR-AUC (Average Precision)']:.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=["Not Spam", "Spam"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))

# --------------------------------------------------
# 6. Compare models
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
).reset_index(drop=True)

results_path = REPORT_DIR / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("VALIDATION MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False, float_format="%.4f"))

print(f"\nComparison saved to: {results_path}")

# --------------------------------------------------
# 7. Save the best validation candidate
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

# Save the vectorizer and model together
artifact = {
    "vectorizer": vectorizer,
    "model": best_model,
    "model_name": best_model_name,
    "positive_label": 1,
    "class_names": {0: "Not Spam", 1: "Spam"},
}

model_path = MODEL_DIR / "spam_classifier_candidate.joblib"
joblib.dump(artifact, model_path)

print("\nBest candidate by validation F1:", best_model_name)
print("Candidate saved to:", model_path)
print("\nClassifier training completed successfully!")
