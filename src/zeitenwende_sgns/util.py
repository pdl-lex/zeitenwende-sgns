import hashlib
import unicodedata


def create_doc_id(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", text).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
