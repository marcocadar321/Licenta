"""
nlp.py – Motor NLP local: preprocesare C++, sentence-transformers, răspunsuri.

Straturi NLP (prioritate descrescătoare):
  1. IntentClassifier cu BERT fine-tunat  (dacă models/ există)
  2. KnowledgeBase cu sentence-transformers (fallback automat)

Dacă biblioteca C++ nu este compilată, normalizarea se face în Python pur.
"""

from __future__ import annotations

import ctypes
import json
import platform
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

import config

_LIB_EXT = ".dll" if platform.system() == "Windows" else ".so"
_LIB_PATH = config.BUILD_DIR / f"preprocess{_LIB_EXT}"



class Preprocessor:
    """
    Normalizează text: lowercasing, elimină caractere speciale, spații multiple.
    Dacă biblioteca C++ nu e disponibilă, folosește o implementare Python identică.
    """

    def __init__(self):
        self._lib = None
        if _LIB_PATH.exists():
            try:
                lib = ctypes.CDLL(str(_LIB_PATH), winmode=0)
                lib.normalize_text.argtypes = [ctypes.c_char_p]
                lib.normalize_text.restype  = ctypes.c_void_p
                lib.free_string.argtypes    = [ctypes.c_void_p]
                lib.free_string.restype     = None
                self._lib = lib
                print(f"  Preprocessor C++ încărcat din {_LIB_PATH}")
            except Exception as exc:
                print(f"  [nlp] Biblioteca C++ nu a putut fi încărcată ({exc}). Fallback Python.")
        else:
            print(
                f"  [nlp] {_LIB_PATH} lipsește. Folosesc normalizare Python.\n"
                "        Compilează cu: g++ -O2 -shared -fPIC -o build/preprocess.so preprocess.cpp"
            )

    def normalize(self, text: str) -> str:
        if self._lib:
            return self._normalize_cpp(text)
        return self._normalize_py(text)

    def _normalize_cpp(self, text: str) -> str:
        ptr = self._lib.normalize_text(text.encode("utf-8"))
        try:
            return ctypes.string_at(ptr).decode("utf-8") if ptr else ""
        finally:
            if ptr:
                self._lib.free_string(ptr)

    @staticmethod
    def _normalize_py(text: str) -> str:
        import re
        text = text.lower()
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        text = re.sub(r" +", " ", text).strip()
        return text



class KnowledgeBase:
    """Indexează intențiile și găsește cel mai bun match prin cosine similarity."""

    def __init__(
        self,
        kb_path:       Path        = config.KB_PATH,
        preprocessor:  Preprocessor = None,
        model_name:    str          = config.NLP_MODEL,
    ):
        from sentence_transformers import SentenceTransformer, util

        self._util         = util
        self._preprocessor = preprocessor or Preprocessor()
        self.intents:       list[dict] = []
        self.patterns:      list[str]  = []
        self._pattern_to_intent: list[int] = []
        self._embeddings   = None

        print(f"  Încarc model sentence-transformers '{model_name}'...", end=" ", flush=True)
        self._model = SentenceTransformer(model_name)
        print("OK")

        self._load(kb_path)
        self._build_index()

    def _load(self, kb_path: Path) -> None:
        with open(kb_path, "r", encoding="utf-8") as f:
            self.intents = json.load(f)["intents"]

    def _build_index(self) -> None:
        for idx, intent in enumerate(self.intents):
            for pattern in intent["patterns"]:
                self.patterns.append(self._preprocessor.normalize(pattern))
                self._pattern_to_intent.append(idx)

        print(f"  Vectorizez {len(self.patterns)} patterne...", end=" ", flush=True)
        self._embeddings = self._model.encode(
            self.patterns, convert_to_tensor=True, show_progress_bar=False
        )
        print("OK")

    def find_best_intent(
        self,
        user_text: str,
        threshold: float = config.NLP_THRESHOLD,
    ) -> tuple[Optional[dict], float]:
        normalized = self._preprocessor.normalize(user_text)
        if not normalized:
            return None, 0.0

        query_emb = self._model.encode(normalized, convert_to_tensor=True)
        scores    = self._util.cos_sim(query_emb, self._embeddings)[0].cpu().numpy()

        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < threshold:
            return None, best_score

        return self.intents[self._pattern_to_intent[best_idx]], best_score




class ResponseEngine:
    """Selectează răspunsuri evitând repetarea celui anterior."""

    FALLBACKS = [
        "Îmi pare rău, nu am înțeles. Poți reformula?",
        "Nu sunt sigur că am înțeles. Încearcă altfel.",
        "Hmm, nu am un răspuns pentru asta. Scrie 'ajutor' pentru sugestii.",
        "Întrebarea ta e în afara domeniului meu actual.",
    ]

    def __init__(self):
        self._last: dict[str, int] = {}

    def get_response(self, intent: dict) -> str:
        tag       = intent["tag"]
        responses = intent["responses"]
        last      = self._last.get(tag, -1)
        choices   = [i for i in range(len(responses)) if i != last] or list(range(len(responses)))
        chosen    = random.choice(choices)
        self._last[tag] = chosen
        return responses[chosen]

    def fallback(self) -> str:
        return random.choice(self.FALLBACKS)




class Chatbot:
    """Punct de intrare principal pentru motorul NLP local."""

    EXIT_COMMANDS = {"exit", "quit", "bye", "pa", "la revedere"}

    def __init__(
        self,
        kb_path:    Path = config.KB_PATH,
        model_name: str  = config.NLP_MODEL,
        threshold:  float = config.NLP_THRESHOLD,
    ):
        self.threshold  = threshold
        self.preprocessor = Preprocessor()

        
        self.kb: KnowledgeBase
        self._nlp_mode: str

        try:
            from bert_classifier import IntentClassifier
            print("Se initializeaza IntentClassifier (BERT)...")
            self.kb        = IntentClassifier(kb_path, self.preprocessor)
            self._nlp_mode = self.kb.mode
        except Exception as exc:
            print(f"  [nlp] IntentClassifier indisponibil ({exc}), folosesc KnowledgeBase.")
            self.kb        = KnowledgeBase(kb_path, self.preprocessor, model_name)
            self._nlp_mode = "sentence_transformers"

        self.engine = ResponseEngine()
        self._stats: dict = {"total": 0, "matched": 0, "time_ms": []}

        print(f"Chatbot gata! {len(self.kb.patterns)} patterne | mod NLP: {self._nlp_mode}")

    

    def process(self, user_input: str) -> Optional[str]:
        """
        Procesează un mesaj. Returnează None dacă userul vrea să iasă.
        """
        text = user_input.strip()
        if not text:
            return "Scrie ceva! :)"
        if text.lower() in self.EXIT_COMMANDS:
            return None

        t0             = time.perf_counter()
        intent, score  = self.kb.find_best_intent(text, self.threshold)
        elapsed_ms     = (time.perf_counter() - t0) * 1000

        self._stats["total"] += 1
        self._stats["time_ms"].append(elapsed_ms)

        if intent:
            self._stats["matched"] += 1
            return self.engine.get_response(intent)
        return self.engine.fallback()

    @property
    def stats(self) -> dict:
        t     = self._stats
        total = t["total"]
        avg   = sum(t["time_ms"]) / len(t["time_ms"]) if t["time_ms"] else 0.0
        return {
            "total":          total,
            "matched":        t["matched"],
            "match_rate_pct": (100 * t["matched"] // total) if total else 0,
            "avg_time_ms":    round(avg, 2),
            "nlp_mode":       self._nlp_mode,
        }
