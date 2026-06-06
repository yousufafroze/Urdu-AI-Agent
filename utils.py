import re

# Common English function words — weak signal but cheap to check
_ENGLISH_MARKERS = frozenset(
    {
        "the",
        "is",
        "are",
        "was",
        "what",
        "how",
        "why",
        "when",
        "please",
        "thanks",
        "thank",
        "hello",
        "hi",
        "hey",
        "can",
        "you",
        "your",
        "my",
        "me",
        "i",
        "we",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
    }
)

_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def has_arabic_script(text: str) -> bool:
    return bool(_ARABIC_SCRIPT_RE.search(text))


def detect_reply_language(text: str) -> str:
    """
    Return 'en' for clearly English input, else 'ur' (Urdu script or Roman Urdu).
    """
    stripped = text.strip()
    if not stripped:
        return "ur"

    if has_arabic_script(stripped):
        return "ur"

    words = {w.lower().strip(".,!?;:'\"") for w in stripped.split()}
    english_hits = len(words & _ENGLISH_MARKERS)
    latin_chars = sum(1 for c in stripped if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in stripped if c.isalpha()) or 1

    # Mostly Latin + several English markers → English
    if latin_chars / total_alpha > 0.85 and english_hits >= 2:
        return "en"

    # Short English questions/commands
    if english_hits >= 1 and len(words) <= 8 and latin_chars / total_alpha > 0.9:
        starters = ("what", "how", "why", "when", "can", "please", "tell", "help")
        first = next(iter(words), "")
        if first in starters:
            return "en"

    return "ur"
