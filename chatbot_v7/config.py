"""
config.py - Toate setarile proiectului, citite din variabile de mediu.

Creaza un fisier .env in radacina proiectului si seteaza valorile de mai jos.
"""

from __future__ import annotations
import os
from pathlib import Path


BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
BUILD_DIR  = BASE_DIR / "build"
STATIC_DIR = BASE_DIR / "static"
KB_PATH    = DATA_DIR / "knowledge_base.json"
USERS_FILE = DATA_DIR / "users.json"


SECRET_KEY:           str = os.getenv("SECRET_KEY", "schimba-asta-cu-o-cheie-secreta-lunga")
TOKEN_EXPIRE_SECONDS: int = int(os.getenv("TOKEN_EXPIRE_SECONDS", "3600"))


RATE_LIMIT_CHAT:  int = int(os.getenv("RATE_LIMIT_CHAT", "30"))
RATE_LIMIT_AUTH:  int = int(os.getenv("RATE_LIMIT_AUTH", "10"))
RATE_WINDOW_SECS: int = 60


MAX_MESSAGE_LENGTH: int = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
MAX_FILE_SIZE_MB:   int = int(os.getenv("MAX_FILE_SIZE_MB", "5"))


MONGODB_URI: str = os.getenv("MONGODB_URI", "")


ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")


NLP_MODEL:     str   = os.getenv("NLP_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
NLP_THRESHOLD: float = float(os.getenv("NLP_THRESHOLD", "0.50"))
