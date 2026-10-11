"""Предклассификация по критериям пользователя (ТЗ п. 4, шаг 1) — семантически, не регулярками.

Критерии — метки с описанием, примерами и (необязательно) ключевыми словами. Откуда берутся:
  * LLM из свободного промпта (`llm_criteria`): «нас оскорбляют?» → метки «оскорбления и угрозы» / «прочее»;
  * структурированный текст пользователя (`parse_criteria`), по строке на метку:
        цифровая услуга: мобильное приложение, звонки в приложении | примеры: не приходит звонок в приложение; дверь из приложения | слова: приложение
        аналоговая услуга: домофон на подъезде, трубка, ключи | примеры: трубка не звонит; заказать ключи | слова: трубка, панель, ключ
Отнесение без LLM (`classify`): прототип метки = среднее эмбеддингов описания, примеров и «псевдо-примеров» — документов,
где встретились ключевые слова (леммы). Документ получает метку с максимальным сходством, если оно ≥ threshold
и отрыв от второй ≥ margin, иначе «прочее». С LLM (`classify_llm`) — батчами JSON; гибрид: LLM только для пограничных.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import text as T


@dataclass
class LabelSpec:
    name: str
    description: str = ""
    examples: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def seed_texts(self) -> list[str]:
        base = f"{self.name}. {self.description}".strip(". ")
        return [base] + list(self.examples)


@dataclass
class Criteria:
    labels: list[LabelSpec]
    other: str = "прочее"
    question: str = ""

    @property
    def names(self) -> list[str]:
        return [l.name for l in self.labels]


def parse_criteria(spec: str, other: str = "прочее") -> Criteria:
    labels = []
    for line in spec.splitlines():
        line = line.strip().lstrip("-*• ")
        if not line or ":" not in line:
            continue
        name, rest = line.split(":", 1)
        parts = [p.strip() for p in rest.split("|")]
        desc, examples, keywords = parts[0], [], []
        for p in parts[1:]:
            low = p.lower()
            if low.startswith(("пример", "examples")):
                examples = [e.strip() for e in re.split(r"[;\n]", p.split(":", 1)[1]) if e.strip()]
            elif low.startswith(("слова", "ключ", "keywords")):
                keywords = [k.strip().lower() for k in re.split(r"[,;]", p.split(":", 1)[1]) if k.strip()]
        labels.append(LabelSpec(name.strip(), desc, examples, keywords))
    return Criteria(labels, other=other)


# ---------- отнесение по эмбеддингам ----------

def _stems(keywords: Sequence[str]) -> tuple[set, tuple]:
    """Точные формы (слово и лемма) и основы длиной ≥ 6 для словообразовательных вариантов (оскорбление ↔ оскорблять).
    Многословные ключи («физическая расправа») разбиваются на слова."""
    exact: set = set()
    for k in keywords:
        for w in T.tokens(k.replace("ё", "е")):
            exact.add(w)
            exact.add(T.lemma(w))
    stems = tuple({w[:6] for w in exact if len(w) >= 7})
    return exact, stems


def _keyword_mask(texts: Sequence[str], keywords: Sequence[str]) -> np.ndarray:
    if not keywords:
        return np.zeros(len(texts), dtype=bool)
    exact, stems = _stems(keywords)
    out = np.zeros(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        toks = set(T.tokens(t))
        lem = toks | {T.lemma(x) for x in toks}
        if lem & exact or (stems and any(l.startswith(stems) for l in lem if len(l) >= 7)):
            out[i] = True
    return out


def label_prototypes(criteria: Criteria, embedder, texts: Optional[Sequence[str]] = None,
                     E: Optional[np.ndarray] = None, max_pseudo: int = 100) -> np.ndarray:
    protos = []
    for lab in criteria.labels:
        vecs = [embedder.encode(lab.seed_texts()).mean(axis=0)]
        if texts is not None and E is not None and lab.keywords:
            m = _keyword_mask(texts, lab.keywords)
            idx = np.where(m)[0][:max_pseudo]
            if len(idx):
                vecs.append(E[idx].mean(axis=0))
        v = np.mean(vecs, axis=0)
        protos.append(v / (np.linalg.norm(v) or 1.0))
    return np.stack(protos)


def classify(E: np.ndarray, criteria: Criteria, protos: np.ndarray, threshold: float = 0.4,
             margin: float = 0.0, texts: Optional[Sequence[str]] = None, keyword_bonus: float = 0.15) -> tuple[list[str], np.ndarray]:
    """Метки и матрица сходств (n × k). Ключевое слово метки в тексте добавляет keyword_bonus к сходству."""
    S = np.asarray(E) @ protos.T
    if texts is not None and keyword_bonus:
        for j, lab in enumerate(criteria.labels):
            if lab.keywords:
                S[:, j] += keyword_bonus * _keyword_mask(texts, lab.keywords)
    names = criteria.names
    out = []
    for row in S:
        order = np.argsort(-row)
        best, second = row[order[0]], (row[order[1]] if len(row) > 1 else -1.0)
        out.append(names[order[0]] if best >= threshold and best - second >= margin else criteria.other)
    return out, S


# ---------- через LLM ----------

CRITERIA_PROMPT = """Пользователь анализирует комментарии сотрудников поддержки к обращениям абонентов и спрашивает: «{question}».
Составь критерии, по которым комментарии отбираются под этот вопрос: 1–4 метки. Для каждой — короткое название,
описание в одну фразу, 2–3 коротких примера комментариев и 3–6 ключевых слов (леммы). Остальное будет «прочее».
Примеры реальных комментариев (стиль, не тема):
{samples}
Ответ строго JSON, без пояснений: {{"labels": [{{"name": "...", "description": "...", "examples": ["..."], "keywords": ["..."]}}]}}"""


def llm_criteria(question: str, llm, samples: Sequence[str] = (), max_samples: int = 12) -> Criteria:
    shown = "\n".join(f"- {s[:120]}" for s in list(samples)[:max_samples]) or "- (нет)"
    data = llm.chat_json([{"role": "user", "content": CRITERIA_PROMPT.format(question=question, samples=shown)}])
    labels = []
    for l in data.get("labels", []) if isinstance(data, dict) else []:
        if isinstance(l, dict) and l.get("name"):
            labels.append(LabelSpec(str(l["name"]), str(l.get("description", "")),
                                    [str(x) for x in l.get("examples", [])], [str(x).lower() for x in l.get("keywords", [])]))
    return Criteria(labels, question=question)


CLASSIFY_SYSTEM = """Ты относишь комментарии сотрудников поддержки к одной из меток. Метки:
{labels}
Если ни одна не подходит — «{other}». Отвечай строго JSON-объектом: номер комментария → название метки."""


def classify_llm(texts: Sequence[str], criteria: Criteria, llm, batch_size: int = 20) -> list[str]:
    desc = "\n".join(f"- {l.name}: {l.description}" + (f" (примеры: {'; '.join(l.examples[:3])})" if l.examples else "")
                     for l in criteria.labels)
    system = CLASSIFY_SYSTEM.format(labels=desc, other=criteria.other)
    valid = set(criteria.names) | {criteria.other}
    out = [criteria.other] * len(texts)
    for k in range(0, len(texts), batch_size):
        batch = texts[k:k + batch_size]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps({str(i + 1): t[:600] for i, t in enumerate(batch)}, ensure_ascii=False)}]
        try:
            data = llm.chat_json(msgs)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for i in range(len(batch)):
            v = data.get(str(i + 1))
            if isinstance(v, str) and v in valid:
                out[k + i] = v
    return out


def classify_hybrid(texts: Sequence[str], E: np.ndarray, criteria: Criteria, protos: np.ndarray, llm,
                    threshold: float = 0.4, uncertain_band: float = 0.1, batch_size: int = 20) -> list[str]:
    """Эмбеддинги решают уверенные случаи, LLM — пограничные (сходство в полосе ±uncertain_band вокруг порога)."""
    labels, S = classify(E, criteria, protos, threshold=threshold, texts=texts)
    best = S.max(axis=1)
    idx = [i for i in range(len(texts)) if abs(best[i] - threshold) <= uncertain_band]
    if idx and llm is not None:
        fixed = classify_llm([texts[i] for i in idx], criteria, llm, batch_size=batch_size)
        for i, v in zip(idx, fixed):
            labels[i] = v
    return labels


def evaluate_labels(pred: Sequence[str], true: Sequence[str]) -> dict:
    pred, true = np.asarray(pred), np.asarray(true)
    out = {"accuracy": float((pred == true).mean()) if len(pred) else 0.0, "per_label": {}}
    for lab in sorted(set(true) | set(pred)):
        tp = int(((pred == lab) & (true == lab)).sum())
        p = int((pred == lab).sum())
        t = int((true == lab).sum())
        out["per_label"][lab] = {"precision": tp / p if p else 0.0, "recall": tp / t if t else 0.0, "support": t}
    return out
