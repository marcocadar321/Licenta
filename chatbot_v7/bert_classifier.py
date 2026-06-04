"""
classifier.py – Clasificator de intentii bazat pe modelul BERT fin-tunat.

Daca modelul fine-tunat nu exista, face fallback automat la KnowledgeBase
(sentence-transformers) din nlp.py.

Utilizare directa:
    from classifier import IntentClassifier
    clf = IntentClassifier(kb_path, preprocessor)
    intent, score = clf.find_best_intent("ce sunt embeddings?")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "intent_classifier"
CONFIDENCE_THRESHOLD = 0.60   
BERT_THRESHOLD       = 0.50   


class BERTClassifier:
    """
    Clasificator bazat pe modelul DistilBERT fin-tunat local.
    Returna intent + probabilitate softmax (nu cosine similarity).
    """

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Modelul fine-tunat nu a fost gasit la: {model_dir}\n"
                "Ruleaza mai intai: python fine_tune.py"
            )

        print(f"  Incarc model BERT fine-tunat din '{model_dir}'...", end=" ", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self._model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self._model.eval()
        print("OK")

       
        label_map_path = model_dir / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"label_map.json lipsa din {model_dir}")
        with open(label_map_path, encoding="utf-8") as f:
            lm = json.load(f)
        self._label2id = lm["label2id"]
        self._id2label = {int(k): v for k, v in lm["id2label"].items()}

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def predict(self, text: str) -> tuple[str, float]:
        """Returneaza (tag_intent, probabilitate)."""
        enc = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)

        with torch.no_grad():
            logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits

        probs    = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        best_idx = int(np.argmax(probs))
        return self._id2label[best_idx], float(probs[best_idx])


class IntentClassifier:
    """
    Facade cu doua straturi:
      1. BERTClassifier (model fine-tunat) – prioritar
      2. KnowledgeBase  (sentence-transformers) – fallback

    Alege automat in functie de ce e disponibil si de scorul de confidenta.
    """

    def __init__(self, kb_path: Path, preprocessor, model_dir: Path = DEFAULT_MODEL_DIR):
        self._kb_path     = kb_path
        self._preprocessor = preprocessor
        self._intents     = self._load_intents(kb_path)

       
        self._bert: BERTClassifier | None = None
        try:
            self._bert = BERTClassifier(model_dir)
            self._mode = "bert_finetuned"
        except FileNotFoundError as e:
            print(f"  [classifier] BERT nu e disponibil: {e}")
            print("  [classifier] Fallback la sentence-transformers.")
            self._mode = "sentence_transformers"

    
        self._st_kb = None
        self._init_st_fallback()

        print(f"  Mod activ: {self._mode}")

    def _load_intents(self, kb_path: Path) -> dict[str, dict]:
        with open(kb_path, encoding="utf-8") as f:
            data = json.load(f)
        return {i["tag"]: i for i in data["intents"]}

    def _init_st_fallback(self):
        """Initializeaza KnowledgeBase (sentence-transformers) pentru fallback."""
        try:
            from nlp import KnowledgeBase
            self._st_kb = KnowledgeBase(self._kb_path, self._preprocessor)
        except Exception as e:
            print(f"  [classifier] sentence-transformers indisponibil: {e}")

    @property
    def mode(self) -> str:
        return self._mode

   
    @property
    def patterns(self) -> list[str]:
        """Compatibilitate cu Chatbot care citeste kb.patterns."""
        if self._st_kb:
            return self._st_kb.patterns
     
        patterns = []
        for intent in self._intents.values():
            patterns.extend(intent.get("patterns", []))
        return patterns

    def find_best_intent(self, user_text: str, threshold: float = BERT_THRESHOLD):
        """
        Cauta cel mai bun intent.
        Returneaza (intent_dict | None, score).
        """
        normalized = self._preprocessor.normalize(user_text)
        if not normalized:
            return None, 0.0

        
        if self._bert is not None:
            tag, prob = self._bert.predict(normalized)

            
            if prob >= CONFIDENCE_THRESHOLD:
                intent = self._intents.get(tag)
                return (intent, prob) if intent else (None, prob)

          
            if prob >= BERT_THRESHOLD and self._st_kb is not None:
                st_intent, st_score = self._st_kb.find_best_intent(user_text, threshold=0.3)
               
                bert_w = prob * 0.6
                st_w   = st_score * 0.4
                if bert_w >= st_w:
                    intent = self._intents.get(tag)
                    return (intent, prob) if intent else (None, prob)
                return st_intent, st_score

            
            if self._st_kb is not None:
                self._mode = "sentence_transformers_fallback"
                return self._st_kb.find_best_intent(user_text, threshold)

            return None, prob

        if self._st_kb is not None:
            return self._st_kb.find_best_intent(user_text, threshold)

        return None, 0.0
