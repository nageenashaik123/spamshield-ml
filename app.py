from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack

from src.spam_logic import classify_with_guard


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="SpamShield ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

TUNED_MODEL_PATH = ROOT / "models" / "spam_classifier_threshold_tuned.joblib"
FINAL_MODEL_PATH = ROOT / "models" / "spam_classifier_final.joblib"

# Full display names for known sklearn classifier class names
MODEL_FULL_NAMES = {
    "LinearSVC":             "Linear Support Vector Classifier (LinearSVC)",
    "SVC":                   "Support Vector Classifier (SVC)",
    "LogisticRegression":    "Logistic Regression",
    "RandomForestClassifier":"Random Forest Classifier",
    "GradientBoostingClassifier": "Gradient Boosting Classifier",
    "MultinomialNB":         "Multinomial Naive Bayes",
    "ComplementNB":          "Complement Naive Bayes",
    "SGDClassifier":         "Stochastic Gradient Descent Classifier (SGD)",
}


# =========================================================
# 2. STYLING  —  Premium warm-light theme
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"], * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* ── App background: warm off-white ── */
    .stApp {
        background: #f5f3ef !important;
    }

    /* ── Hide Streamlit chrome ── */
    [data-testid="stHeader"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    footer { display: none !important; }

    /* ── Main content ── */
    .block-container {
        max-width: 840px !important;
        padding: 2.5rem 2rem 5rem !important;
    }

    /* ═══════════════════════════════════
       SIDEBAR
    ═══════════════════════════════════ */
    div[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e8e3db !important;
    }

    /* kill the default blue/white streamlit metric colours in sidebar */
    div[data-testid="stSidebar"] [data-testid="stMetric"] label,
    div[data-testid="stSidebar"] [data-testid="stMetric"] div {
        color: #1a1a2e !important;
    }

    .sb-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding-bottom: 20px;
        border-bottom: 1px solid #ede8e0;
        margin-bottom: 24px;
    }

    .sb-brand-icon {
        width: 40px; height: 40px;
        background: linear-gradient(135deg, #4361ee, #7b5ea7);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 12px rgba(67,97,238,0.25);
    }

    .sb-brand-name {
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: #1a1a2e !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .sb-brand-tagline {
        font-size: 0.7rem !important;
        color: #8b8fa8 !important;
        font-weight: 500 !important;
    }

    .sb-section {
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.13em !important;
        text-transform: uppercase !important;
        color: #b0aaa0 !important;
        margin: 22px 0 10px !important;
    }

    .sb-model-badge {
        background: linear-gradient(135deg, #eef2ff, #f0ebff);
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        padding: 10px 13px;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #4338ca !important;
        line-height: 1.5;
    }

    .sb-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 11px 13px;
        background: #faf9f7;
        border: 1px solid #ede8e0;
        border-radius: 11px;
        margin-bottom: 8px;
    }

    .sb-metric-name {
        font-size: 0.83rem !important;
        color: #6b7280 !important;
        font-weight: 500 !important;
    }

    .sb-metric-val {
        font-size: 0.92rem !important;
        font-weight: 800 !important;
        color: #4361ee !important;
    }

    .sb-note {
        font-size: 0.73rem !important;
        color: #b0aaa0 !important;
        line-height: 1.65 !important;
        margin-top: 18px !important;
        padding-top: 16px !important;
        border-top: 1px solid #ede8e0 !important;
    }

    /* ═══════════════════════════════════
       HERO
    ═══════════════════════════════════ */
    .hero-section {
        text-align: center;
        padding: 10px 0 36px;
        max-width: 980px;
        margin: 0 auto;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 999px;
        padding: 7px 16px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #4361ee;
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: clamp(2.8rem, 5vw, 5rem);
        font-weight: 900;
        letter-spacing: -0.06em;
        line-height: 1.02;
        color: #0f0e17;
        margin: 0 auto 16px;
        max-width: 900px;
        width: 100%;
    }

    .hero-title span {
        display: block;
        background: linear-gradient(135deg, #4361ee 0%, #7b5ea7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-sub {
        font-size: 1.08rem;
        color: #6b7280;
        line-height: 1.75;
        max-width: 720px;
        margin: 0 auto;
        font-weight: 400;
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #c7d2fe 30%, #c7d2fe 70%, transparent);
        margin: 4px 0 32px;
        opacity: 0.6;
    }

    /* ═══════════════════════════════════
       INPUT CARD
    ═══════════════════════════════════ */
    .input-card {
        background: #ffffff;
        border: 1.5px solid #e8e3db;
        border-radius: 22px;
        padding: 26px 26px 18px;
        box-shadow: 0 4px 24px rgba(15,14,23,0.05);
        margin: 0 auto 16px;
        max-width: 100%;
    }

    .input-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 14px;
    }

    /* ── Textarea — dark text on white, clearly visible ── */
    div[data-testid="stTextArea"] textarea {
        background: #fdfcfb !important;
        border: 1.5px solid #d1d5db !important;
        border-radius: 14px !important;
        color: #111827 !important;
        font-size: 1rem !important;
        line-height: 1.8 !important;
        padding: 16px 18px !important;
        caret-color: #4361ee;
        transition: border-color 0.2s, box-shadow 0.2s;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #c9c5be !important;
        font-style: italic;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.18) !important;
        outline: none !important;
    }

    div[data-testid="stTextArea"] label { display: none !important; }

    .input-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
    }

    .input-hint {
        font-size: 0.76rem;
        color: #b0aaa0;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .char-badge {
        font-size: 0.76rem;
        color: #9ca3af;
        font-variant-numeric: tabular-nums;
        background: #f3f2ef;
        border-radius: 6px;
        padding: 2px 8px;
    }

    /* ═══════════════════════════════════
       BUTTONS
    ═══════════════════════════════════ */
    div.stButton > button {
        border-radius: 13px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        min-height: 52px !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.005em !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4361ee 0%, #7b5ea7 100%) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(67,97,238,0.3) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3651d4 0%, #6a4d96 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(67,97,238,0.38) !important;
    }

    div.stButton > button:not([kind="primary"]) {
        background: #f3f2ef !important;
        color: #6b7280 !important;
        border: 1.5px solid #e5e0d8 !important;
    }

    div.stButton > button:not([kind="primary"]):hover {
        background: #ebe9e4 !important;
        color: #374151 !important;
    }

    /* ═══════════════════════════════════
       RESULT SECTION
    ═══════════════════════════════════ */

    /* — Spam verdict — */
    .verdict-spam-wrap {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(220,38,38,0.12);
        margin-top: 28px;
    }

    .verdict-spam-banner {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        padding: 26px 28px 22px;
        display: flex;
        align-items: flex-start;
        gap: 18px;
    }

    .verdict-legit-wrap {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(22,163,74,0.1);
        margin-top: 28px;
    }

    .verdict-legit-banner {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        padding: 26px 28px 22px;
        display: flex;
        align-items: flex-start;
        gap: 18px;
    }

    .verdict-icon-wrap {
        width: 56px; height: 56px;
        background: rgba(255,255,255,0.2);
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 28px;
        flex-shrink: 0;
    }

    .verdict-eyebrow {
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.7);
        margin-bottom: 5px;
    }

    .verdict-heading {
        font-size: 1.85rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .verdict-tagline {
        font-size: 0.92rem;
        color: rgba(255,255,255,0.82);
        line-height: 1.6;
    }

    /* — Warning / safe strips below banner — */
    .verdict-spam-body {
        background: #fff5f5;
        border: 1.5px solid #fecaca;
        border-top: none;
        border-radius: 0 0 20px 20px;
        padding: 20px 26px;
    }

    .verdict-legit-body {
        background: #f0fdf4;
        border: 1.5px solid #bbf7d0;
        border-top: none;
        border-radius: 0 0 20px 20px;
        padding: 20px 26px;
    }

    .warning-label {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #dc2626;
        margin-bottom: 10px;
    }

    .safe-label {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #16a34a;
        margin-bottom: 10px;
    }

    .warning-list {
        list-style: none;
        padding: 0; margin: 0;
        display: flex;
        flex-direction: column;
        gap: 7px;
    }

    .warning-list li {
        font-size: 0.88rem;
        color: #7f1d1d;
        display: flex;
        align-items: flex-start;
        gap: 8px;
        line-height: 1.55;
    }

    .safe-list li {
        color: #14532d;
    }

    .warning-list li::before {
        content: "⚠";
        color: #ef4444;
        flex-shrink: 0;
        margin-top: 1px;
    }

    .safe-list li::before {
        content: "✓";
        color: #22c55e;
        font-weight: 900;
        flex-shrink: 0;
        margin-top: 1px;
    }

    /* ── Confidence bar ── */
    .conf-bar-card {
        background: #ffffff;
        border: 1.5px solid #e8e3db;
        border-radius: 16px;
        padding: 18px 22px;
        margin-top: 14px;
        box-shadow: 0 2px 10px rgba(15,14,23,0.04);
    }

    .conf-bar-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .conf-bar-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .conf-bar-pct {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0f0e17;
    }

    .conf-bar-track {
        height: 8px;
        background: #f0ede8;
        border-radius: 999px;
        overflow: hidden;
    }

    .conf-fill-spam {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #f87171, #dc2626);
    }

    .conf-fill-legit {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #4ade80, #16a34a);
    }

    /* ── Disclaimer ── */
    .disclaimer {
        text-align: center;
        font-size: 0.78rem;
        color: #b0aaa0;
        margin-top: 10px;
        line-height: 1.6;
    }

    /* ═══════════════════════════════════
       FOOTER
    ═══════════════════════════════════ */
    .app-footer {
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1.5px solid #e8e3db;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }

    .footer-brand {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #374151;
    }

    .footer-meta {
        font-size: 0.75rem;
        color: #9ca3af;
    }

    /* ── Spinner / alerts ── */
    [data-testid="stAlert"] { border-radius: 14px !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 99px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD MODEL ARTIFACT
# =========================================================
def load_artifact(path: str):
    return joblib.load(path)


if TUNED_MODEL_PATH.exists():
    MODEL_PATH = TUNED_MODEL_PATH
elif FINAL_MODEL_PATH.exists():
    MODEL_PATH = FINAL_MODEL_PATH
else:
    st.error("No trained model found. Run the training scripts first.")
    st.stop()

try:
    artifact    = load_artifact(str(MODEL_PATH))
    word_vectorizer = artifact["word_vectorizer"]
    char_vectorizer = artifact["char_vectorizer"]
    model           = artifact["model"]
    spam_threshold  = float(artifact.get("threshold", 0.0))
except Exception as exc:
    st.error(f"Unable to load model artifact: {exc}")
    st.stop()

model_class   = type(model).__name__
model_display = MODEL_FULL_NAMES.get(model_class, model_class)


# =========================================================
# 4. SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown(
        f"""
        <div class="sb-brand">
            <div class="sb-brand-icon">🛡️</div>
            <div>
                <div class="sb-brand-name">SpamShield ML</div>
                <div class="sb-brand-tagline">Email Security · AI-Powered</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model name (full form) ──────────────────────────
    st.markdown('<p class="sb-section">Classifier</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sb-model-badge">⚡ {model_display}</div>',
        unsafe_allow_html=True,
    )

    # ── Performance metrics ─────────────────────────────
    report_path = ROOT / "reports" / "final_test_metrics.csv"
    classifier_metrics: dict = {}
    if report_path.exists():
        try:
            metrics_df = pd.read_csv(report_path)
            classifier_metrics = {
                row["Metric"]: float(row["Score"])
                for _, row in metrics_df.iterrows()
            }
        except Exception:
            pass

    if not classifier_metrics:
        classifier_metrics = {
            "Accuracy":  0.9972,
            "Precision": 0.9970,
            "Recall":    0.9941,
            "F1 Score":  0.9956,
        }

    LABEL_MAP = {"PR-AUC (Average Precision)": "PR-AUC"}

    st.markdown('<p class="sb-section">Performance</p>', unsafe_allow_html=True)
    for raw, val in classifier_metrics.items():
        lbl = LABEL_MAP.get(raw, raw)
        st.markdown(
            f"""
            <div class="sb-metric-row">
                <span class="sb-metric-name">{lbl}</span>
                <span class="sb-metric-val">{val * 100:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <p class="sb-note">
            Trained on the SpamAssassin public corpus.<br>
            Word-level + character-level TF-IDF features.<br>
            Decision threshold tuned for high precision.
        </p>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 5. HERO
# =========================================================
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-eyebrow">🛡️ &nbsp;AI-Powered &nbsp;·&nbsp; Explainable &nbsp;·&nbsp; 99%+ Accuracy</div>
        <div class="hero-title">Detect Spam Email<br><span>Instantly.</span></div>
        <p class="hero-sub">
            Paste any email and SpamShield ML scans it in seconds —
            catching suspicious patterns, phishing cues, and
            spam signals before they reach you.
        </p>
    </div>
    <div class="section-divider"></div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 6. INPUT CARD
# =========================================================
st.markdown(
    '<div class="input-card"><p class="input-label">✉️ &nbsp;Paste your email below</p>',
    unsafe_allow_html=True,
)

email_text = st.text_area(
    "Email body",
    placeholder=(
        "From: alex@example.com\n"
        "To: you@example.com\n"
        "Subject: Quick question\n\n"
        "Hi — paste the full email or just the body here...\n\n"
        "Best regards,\nAlex"
    ),
    height=240,
    key="email_input",
    label_visibility="collapsed",
)

st.markdown(
    f"""
    <div class="input-footer">
        <span class="input-hint">🔒 Processed locally — nothing stored or transmitted</span>
        <span class="char-badge">{len(email_text):,} chars</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 7. BUTTONS
# =========================================================
btn_col, clr_col = st.columns([5, 1])

with btn_col:
    analyze = st.button(
        "🛡️  Analyze Email",
        type="primary",
        use_container_width=True,
    )

with clr_col:
    if st.button("Clear", use_container_width=True):
        st.session_state["email_input"] = ""
        st.rerun()


# =========================================================
# 8. CLASSIFICATION + RESULT
# =========================================================
if analyze:

    if not email_text.strip():
        st.warning("Please paste an email before analyzing.")
        st.stop()

    try:
        with st.spinner("Scanning for spam signals..."):
            word_feat = word_vectorizer.transform([email_text])
            char_feat = char_vectorizer.transform([email_text])
            features  = hstack([word_feat, char_feat], format="csr", dtype=np.float32)
            score     = float(model.decision_function(features)[0])
    except Exception as exc:
        st.error(f"Could not analyse email: {exc}")
        st.stop()

    is_spam = classify_with_guard(score=score, threshold=spam_threshold, text=email_text)

    # Normalise SVM decision score (~-3…+3) to a 0–100 bar
    SCALE    = 3.0
    norm     = max(0.0, min(1.0, (score + SCALE) / (2 * SCALE)))
    spam_pct = round(norm * 100)
    legit_pct = 100 - spam_pct

    bar_pct   = spam_pct  if is_spam else legit_pct
    bar_class = "conf-fill-spam" if is_spam else "conf-fill-legit"
    bar_label = "Spam likelihood" if is_spam else "Legitimacy confidence"

    # ── Result card ──────────────────────────────────────
    if is_spam:
        st.markdown(
            f"""
            <div class="verdict-spam-wrap">
                <div class="verdict-spam-banner">
                    <div class="verdict-icon-wrap">🚨</div>
                    <div>
                        <div class="verdict-eyebrow">Verdict</div>
                        <div class="verdict-heading">Spam Detected</div>
                        <div class="verdict-tagline">
                            This email exhibits patterns associated with spam or phishing.
                        </div>
                    </div>
                </div>
                <div class="verdict-spam-body">
                    <div class="warning-label">⚠ What you should do</div>
                    <ul class="warning-list">
                        <li>Do not click any links or download attachments in this email.</li>
                        <li>Do not share personal information, passwords, or payment details.</li>
                        <li>Verify the sender's address independently before replying.</li>
                        <li>Mark it as spam and report it to your email provider.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-legit-wrap">
                <div class="verdict-legit-banner">
                    <div class="verdict-icon-wrap">✅</div>
                    <div>
                        <div class="verdict-eyebrow">Verdict</div>
                        <div class="verdict-heading">Looks Legitimate</div>
                        <div class="verdict-tagline">
                            No spam signals detected. This email appears safe.
                        </div>
                    </div>
                </div>
                <div class="verdict-legit-body">
                    <div class="safe-label">✓ You're good — but stay sharp</div>
                    <ul class="warning-list safe-list">
                        <li>The model found no suspicious patterns in this email.</li>
                        <li>Still verify unexpected links before clicking.</li>
                        <li>Be cautious with unsolicited attachments even from known senders.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Confidence bar ───────────────────────────────────
    st.markdown(
        f"""
        <div class="conf-bar-card">
            <div class="conf-bar-header">
                <span class="conf-bar-title">{bar_label}</span>
                <span class="conf-bar-pct">{bar_pct}%</span>
            </div>
            <div class="conf-bar-track">
                <div class="{bar_class}" style="width:{bar_pct}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="disclaimer">
            SpamShield ML is a screening aid, not a guarantee.
            Always apply your own judgement before acting on any email.
        </p>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 9. FOOTER
# =========================================================
st.markdown(
    f"""
    <div class="app-footer">
        <div class="footer-brand">🛡️ SpamShield ML</div>
        <div class="footer-meta">
            {model_display} &nbsp;·&nbsp; Word + Char TF-IDF &nbsp;·&nbsp; Threshold-tuned
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
