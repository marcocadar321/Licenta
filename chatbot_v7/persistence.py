"""
persistence.py – Stocare conversații în MongoDB Atlas (opțional).

Dacă MONGODB_URI nu este setat, toate operațiile sunt silențios ignorate
(chatbot-ul funcționează normal fără bază de date).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import config


_collection = None
_initialized = False


def _get_collection():
    global _collection, _initialized
    if _initialized:
        return _collection  

    _initialized = True
    if not config.MONGODB_URI:
        return None

    try:
        from pymongo import MongoClient

        client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5_000)
        client.admin.command("ping")
        db = client["chatbot"]
        _collection = db["conversations"]
        _collection.create_index("timestamp")
        print("[persistence] MongoDB conectat.")
    except Exception as exc:
        print(f"[persistence] MongoDB indisponibil: {exc}. Continuăm fără persistență.")
        _collection = None

    return _collection




def log_conversation(
    user_message: str,
    bot_response:  str,
    matched:       bool,
    score:         float,
) -> Optional[str]:
    """Salvează o conversație. Returnează ID-ul documentului sau None."""
    col = _get_collection()
    if col is None:
        return None

    try:
        result = col.insert_one({
            "timestamp":    datetime.now(timezone.utc),
            "user":         user_message,
            "bot":          bot_response,
            "matched":      matched,
            "score":        score,
        })
        return str(result.inserted_id)
    except Exception as exc:
        print(f"[persistence] Nu am putut salva conversația: {exc}")
        return None


def get_recent_conversations(limit: int = 20) -> list[dict]:
    """Returnează ultimele `limit` conversații."""
    col = _get_collection()
    if col is None:
        return []
    try:
        docs = col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        return list(docs)
    except Exception as exc:
        print(f"[persistence] {exc}")
        return []
