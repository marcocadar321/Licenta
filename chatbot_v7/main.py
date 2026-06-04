"""
main.py - Interfata terminal pentru chatbot.

Rulare:
    python main.py
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

from nlp import Chatbot 


def main() -> None:
    print("=" * 45)
    print("  ChatBot NLP v7.0")
    print("  Mod: NLP local")
    print("  Tasteaza 'exit' pentru a iesi.")
    print("=" * 45)

    bot = Chatbot()

    while True:
        try:
            user_input = input("\nTu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nLa revedere!")
            break

        if not user_input:
            continue

        response = bot.process(user_input)
        if response is None:
            print("La revedere!")
            break

        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
