"""
security.py – Autentificare JWT, rate limiting, sanitizare input.

Nu importă nimic din restul proiectului → poate fi testat izolat.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

# ── State în memorie pentru rate limiting ─────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)


# ── Parole ────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return f"{salt}:{h.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, h = stored_hash.split(":", 1)
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
        return hmac.compare_digest(expected.hex(), h)
    except Exception:
        return False


# ── JWT simplu (HMAC-SHA256 + base64url) ──────────────────────────────────────

def create_token(username: str) -> str:
    expires = int(time.time()) + config.TOKEN_EXPIRE_SECONDS
    payload = f"{username}:{expires}"
    sig = hmac.new(config.SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}:{sig}".encode()).decode()


def decode_token(token: str) -> Optional[str]:
    """Returnează username dacă tokenul e valid, altfel None."""
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        username, expires_str, sig = raw.rsplit(":", 2)
        payload = f"{username}:{expires_str}"
        expected = hmac.new(config.SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return None
        if int(time.time()) > int(expires_str):
            return None
        return username
    except Exception:
        return None


# ── Utilizatori ──────────────────────────────────────────────────────────────

def _load_users() -> dict:
    config.USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not config.USERS_FILE.exists():
        config.USERS_FILE.write_text("{}", encoding="utf-8")
    try:
        return json.loads(config.USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    config.USERS_FILE.write_text(
        json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if not re.match(r"^[a-z0-9_]{3,32}$", username):
        return False, "Userul trebuie să aibă 3-32 caractere (litere, cifre, underscore)."
    if len(password) < 8:
        return False, "Parola trebuie să aibă minim 8 caractere."
    if len(password) > 128:
        return False, "Parola prea lungă."

    users = _load_users()
    if username in users:
        return False, "Userul există deja."

    users[username] = {
        "password_hash": hash_password(password),
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "last_login":    None,
    }
    _save_users(users)
    return True, "Cont creat cu succes."


def login_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    users = _load_users()
    user = users.get(username)
    if not user or not verify_password(password, user["password_hash"]):
        return False, "User sau parolă incorectă."
    users[username]["last_login"] = datetime.now(timezone.utc).isoformat()
    _save_users(users)
    return True, create_token(username)




def check_rate_limit(key: str, limit: int) -> tuple[bool, int]:
    """
    Returnează (allowed, retry_after_seconds).
    Folosește o fereastră glisantă de RATE_WINDOW_SECS secunde.
    """
    now          = time.time()
    window_start = now - config.RATE_WINDOW_SECS
    calls = [t for t in _rate_store[key] if t > window_start]
    _rate_store[key] = calls

    if len(calls) >= limit:
        retry_after = int(config.RATE_WINDOW_SECS - (now - calls[0])) + 1
        return False, retry_after

    _rate_store[key].append(now)
    return True, 0




def sanitize_input(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"<script[\s\S]*?>[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"javascript\s*:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"on\w+\s*=", "", text, flags=re.IGNORECASE)
    return text


def validate_message(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Mesajul nu poate fi gol."
    if len(text) > config.MAX_MESSAGE_LENGTH:
        return False, f"Mesajul depășește {config.MAX_MESSAGE_LENGTH} caractere."
    return True, ""


def validate_file(filename: str, size_bytes: int, content_type: str) -> tuple[bool, str]:
    allowed_extensions = {
        ".jpg", ".jpeg", ".png", ".gif", ".webp",
        ".txt", ".md", ".py", ".js", ".ts", ".json",
        ".csv", ".html", ".css", ".cpp", ".c", ".java",
        ".pdf", ".xml", ".yaml", ".yml",
    }
    allowed_mime_prefixes = (
        "image/", "text/", "application/json", "application/pdf", "application/xml",
    )

    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        return False, f"Extensia '{ext}' nu este permisă."

    if content_type and not any(content_type.startswith(p) for p in allowed_mime_prefixes):
        return False, f"Tipul de fișier '{content_type}' nu este permis."

    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        return False, f"Fișierul depășește {config.MAX_FILE_SIZE_MB} MB."

    return True, ""
