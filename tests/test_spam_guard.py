from src.spam_logic import classify_with_guard, looks_like_legitimate_email


def test_legitimate_email_is_not_forced_to_spam():
    email = (
        "Hi Sarah,\n\nThanks for the meeting notes. "
        "I will review the project plan and send feedback by Friday afternoon. "
        "Please share the updated timeline when you have it.\n\nBest,\nAlex"
    )

    assert looks_like_legitimate_email(email) is True
    assert classify_with_guard(score=0.304821, threshold=-0.056042, text=email) is False


def test_obvious_spam_stays_spam():
    email = (
        "URGENT! You have won a free vacation and must click now to claim your prize. "
        "We need your banking details immediately."
    )

    assert classify_with_guard(score=0.85, threshold=-0.056042, text=email) is True
