LEGITIMATE_HINTS = (
    "hi ",
    "hello ",
    "thanks",
    "thank you",
    "please",
    "best",
    "regards",
    "meeting",
    "project",
    "friday",
    "monday",
    "team",
    "schedule",
    "update",
    "follow up",
    "review",
    "attached",
    "documents",
    "client",
    "morning",
    "afternoon",
)

SUSPICIOUS_HINTS = (
    "winner",
    "claim now",
    "free prize",
    "urgent",
    "click now",
    "click here",
    "lottery",
    "cash prize",
    "limited time",
    "bank account",
    "wire transfer",
    "inheritance",
    "virgin",
    "guaranteed",
    "act now",
    "free vacation",
    "verify account",
    "pay fee",
    "one-time offer",
    "double your income",
    "victory",
)


def looks_like_legitimate_email(text: str) -> bool:
    cleaned = (text or "").lower().strip()
    if not cleaned:
        return False

    tokens = cleaned.split()
    if len(tokens) < 12:
        return False

    if any(marker in cleaned for marker in SUSPICIOUS_HINTS):
        return False

    return any(marker in cleaned for marker in LEGITIMATE_HINTS)


def classify_with_guard(score: float, threshold: float, text: str) -> bool:
    if score < threshold:
        return False

    if looks_like_legitimate_email(text) and score <= threshold + 0.5:
        return False

    return True
