"""normalize_text op — text preprocessing."""

import re

from hush.core.ops import op


@op
def normalize_text(customer_speech: str):
    """Lowercase, remove special chars, collapse whitespace."""
    text = customer_speech.lower()
    text = re.sub(r'[^\w\s\+]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return {"normalized_text": text}
