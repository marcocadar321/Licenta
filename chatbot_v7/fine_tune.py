"""
fine_tune.py – Fine-tuneaza distilbert-base-multilingual-cased
pe intentiile din knowledge_base.json.

Utilizare:
    python fine_tune.py                        # antrenare standard
    python fine_tune.py --epochs 10            # mai multe epoci
    python fine_tune.py --output models/bert   # director output custom
    python fine_tune.py --augment              # cu data augmentation

Output:
    models/intent_classifier/   <- model fine-tunat + tokenizer + label map
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)



BASE_MODEL   = "distilbert-base-multilingual-cased"
DATA_PATH    = Path(__file__).parent / "data" / "knowledge_base.json"
DEFAULT_OUT  = Path(__file__).parent / "models" / "intent_classifier"

EPOCHS       = 8
BATCH_SIZE   = 8
LR           = 3e-5
MAX_LEN      = 64
WARMUP_RATIO = 0.1
SEED         = 42


_SYNONYMS = {
    "salut": ["hey", "buna", "alo"],
    "buna": ["salut", "hey", "hello"],
    "multumesc": ["mersi", "multam", "thanks"],
    "cum": ["in ce mod", "de ce"],
    "este": ["e", "reprezinta"],
    "folosesti": ["utilizezi", "aplici"],
    "functioneaza": ["merge", "lucreaza"],
}

def augment(text: str, n: int = 2) -> list[str]:
    """Genereaza n variante prin inlocuire aleatorie de sinonime."""
    results = []
    words = text.split()
    for _ in range(n):
        new = [random.choice(_SYNONYMS.get(w.lower(), [w])) for w in words]
        candidate = " ".join(new)
        if candidate != text:
            results.append(candidate)
    return results




class IntentDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int = MAX_LEN,
    ):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }




def load_data(path: Path, augment_data: bool = False):
    with open(path, encoding="utf-8") as f:
        kb = json.load(f)

    label2id = {intent["tag"]: idx for idx, intent in enumerate(kb["intents"])}
    id2label = {v: k for k, v in label2id.items()}

    texts, labels = [], []
    for intent in kb["intents"]:
        idx = label2id[intent["tag"]]
        for pattern in intent["patterns"]:
            texts.append(pattern)
            labels.append(idx)
            if augment_data:
                for aug in augment(pattern):
                    texts.append(aug)
                    labels.append(idx)

    return texts, labels, label2id, id2label


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_val_split(texts, labels, val_ratio=0.15):
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    split = max(1, int(len(combined) * val_ratio))
    val, train = combined[:split], combined[split:]
    t_texts, t_labels = zip(*train)
    v_texts, v_labels = zip(*val)
    return list(t_texts), list(t_labels), list(v_texts), list(v_labels)



def train(
    epochs: int = EPOCHS,
    output_dir: Path = DEFAULT_OUT,
    augment_data: bool = False,
):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    
    print("Incarc datele...", end=" ", flush=True)
    texts, labels, label2id, id2label = load_data(DATA_PATH, augment_data)
    t_texts, t_labels, v_texts, v_labels = train_val_split(texts, labels)
    print(f"OK  (train={len(t_texts)}, val={len(v_texts)}, clase={len(label2id)})")

    
    print(f"Incarc '{BASE_MODEL}'...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)
    print("OK")

    
    train_ds = IntentDataset(t_texts, t_labels, tokenizer)
    val_ds   = IntentDataset(v_texts, v_labels, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps   = len(train_dl) * epochs
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    
    best_val_acc = 0.0
    print(f"\n{'─'*52}")
    print(f"  Epoca  | Train Loss | Val Loss | Val Acc")
    print(f"{'─'*52}")

    for epoch in range(1, epochs + 1):
        
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids      = batch["label"].to(device)

            optimizer.zero_grad()
            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dl)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_dl:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label_ids      = batch["label"].to(device)

                out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
                val_loss += out.loss.item()
                preds     = out.logits.argmax(dim=-1)
                correct  += (preds == label_ids).sum().item()
                total    += len(label_ids)

        avg_val_loss = val_loss / len(val_dl)
        val_acc      = correct / total if total else 0.0

        marker = " ★" if val_acc > best_val_acc else ""
        print(f"  {epoch:^5}  | {avg_train_loss:^10.4f} | {avg_val_loss:^8.4f} | {val_acc*100:^6.1f}%{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

    label_map = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"{'─'*52}")
    print(f"  Best val acc: {best_val_acc*100:.1f}%")
    print(f"  Model salvat in: {output_dir}")
    return str(output_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT pe intentii.")
    parser.add_argument("--epochs",  type=int,  default=EPOCHS,      help="Numar epoci")
    parser.add_argument("--output",  type=str,  default=str(DEFAULT_OUT), help="Director output")
    parser.add_argument("--augment", action="store_true", help="Activeaza data augmentation")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        output_dir=Path(args.output),
        augment_data=args.augment,
    )
