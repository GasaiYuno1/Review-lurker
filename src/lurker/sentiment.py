"""Тональность комментария готовой открытой моделью (Hugging Face, без обучения): вероятности [негатив, нейтрально,
позитив]. Проверка (research/comments/sentiment.py): на отзывах клиентов доля негатива по кластерам повторяет настоящую
(blanchefort/rubert-base-cased-sentiment: корреляция 0.99, ошибка 4 п.п. на отзывах о банках); на заметках операторов
(«абонент недоволен, угрожает…») модели, обученные на отзывах, ошибаются — для тикетов тональность кластера лучше брать
у LLM из описания кластера."""
from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_MODEL = "blanchefort/rubert-base-cased-sentiment"
CLASSES = ("neg", "neu", "pos")


def _canon(label: str) -> str:
    l = label.lower()
    return "neg" if l.startswith("neg") else "pos" if l.startswith("pos") else "neu"


def predict_sentiment(texts: Sequence[str], model: str = DEFAULT_MODEL, batch: int = 32) -> np.ndarray:
    """→ матрица (n, 3): вероятности негатив / нейтрально / позитив. Тексты длиннее 512 токенов обрезаются."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForSequenceClassification.from_pretrained(model).eval()
    idx = {i: CLASSES.index(_canon(l)) for i, l in m.config.id2label.items()}
    texts = [t if t and t.strip() else "-" for t in texts]
    order = np.argsort([len(t) for t in texts])
    P = np.zeros((len(texts), 3), dtype=np.float32)
    with torch.inference_mode():
        for a in range(0, len(texts), batch):
            ids = order[a:a + batch]
            enc = tok([texts[i] for i in ids], padding=True, truncation=True, max_length=512, return_tensors="pt")
            p = torch.softmax(m(**enc).logits, dim=-1).numpy()
            for j, i in enumerate(ids):
                for c, v in enumerate(p[j]):
                    P[i, idx[c]] += v
    return P
