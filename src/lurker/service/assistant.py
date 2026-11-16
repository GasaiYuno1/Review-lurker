"""Ассистент по выборке (ТЗ п. 2, 6.2): вопрос → период из текста вопроса (фильтр до анализа) → отбор релевантных
обращений (эмбеддинги + совпадение лемм) → ответ.
Без LLM: сколько похожих обращений и когда (по периодам), в каких темах, цитаты с датами.
С LLM: те же цитаты и счётчики в промпт, ответ с опорой на них (RAG). Сравнение периодов — через summarize.llm_compare.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .. import text as T
from ..dynamics import FREQ, apply_filters, compare_periods
from ..period import Period, describe, parse_periods


@dataclass
class Answer:
    text: str
    quotes: list[dict] = field(default_factory=list)      # {date, text, topic, score}
    counts: Optional[pd.Series] = None                     # релевантные обращения по периодам
    by_topic: Optional[pd.Series] = None
    periods: list[Period] = field(default_factory=list)
    n_relevant: int = 0
    n_selected: int = 0
    compare: Optional[pd.DataFrame] = None                 # сравнение двух периодов по темам (среди релевантных)
    quotes_by_period: dict = field(default_factory=dict)   # label → список цитат


def retrieve(question: str, df: pd.DataFrame, E: np.ndarray, embedder, k: int = 20, periods: Sequence[Period] = (),
             filters: Optional[dict] = None, topic: Optional[int] = None, keyword_weight: float = 0.3,
             text_col: str = "clean") -> pd.DataFrame:
    """Строки df с колонкой score по убыванию. E выровнена с позициями df (RangeIndex)."""
    d = apply_filters(df, filters, periods)
    if topic is not None and "topic" in d.columns:
        d = d[d["topic"] == topic]
    d = d[d[text_col].str.len() > 3]
    if d.empty:
        return d.assign(score=[])
    q = embedder.encode([question])[0]
    pos = d.index.to_numpy()
    score = E[pos] @ q
    ql = set(T.lemmas(question))
    if ql and keyword_weight:
        overlap = np.array([len(ql & set(T.lemmas(t))) / len(ql) for t in d[text_col]])
        score = score + keyword_weight * overlap
    out = d.assign(score=score).sort_values("score", ascending=False)
    return out.head(k) if k else out


def relevant_set(question: str, df: pd.DataFrame, E: np.ndarray, embedder, periods: Sequence[Period] = (),
                 filters: Optional[dict] = None, min_score: float = 0.45, max_n: int = 2000) -> pd.DataFrame:
    """Все обращения, похожие на вопрос (для счётчиков «как часто»): score ≥ min_score, не более max_n."""
    r = retrieve(question, df, E, embedder, k=0, periods=periods, filters=filters)
    return r[r["score"] >= min_score].head(max_n)


def answer(question: str, df: pd.DataFrame, E: np.ndarray, embedder, llm=None, k: int = 15, freq: str = "M",
           filters: Optional[dict] = None, names: Optional[dict] = None, min_score: float = 0.45,
           use_llm_periods: bool = False) -> Answer:
    ref = df["date"].max().date() if df["date"].notna().any() else date.today()
    periods: list[Period] = []
    if use_llm_periods and llm is not None:
        from ..period import llm_periods
        try:
            periods = llm_periods(question, ref, llm, df["date"].min().date(), ref)
        except Exception:
            periods = []
    if not periods:
        periods = parse_periods(question, ref)
    selected = apply_filters(df, filters, periods)
    rel = relevant_set(question, df, E, embedder, periods, filters, min_score=min_score)
    top = rel.head(k) if len(rel) >= k else retrieve(question, df, E, embedder, k=k, periods=periods, filters=filters)
    names = names or {}
    quotes = [{"date": r["date"].strftime("%d.%m.%Y") if pd.notna(r["date"]) else "", "text": r["clean"],
               "topic": names.get(r.get("topic"), str(r.get("topic", ""))), "score": float(r["score"])} for _, r in top.iterrows()]
    counts = rel.groupby(pd.Grouper(key="date", freq=FREQ.get(freq, freq))).size().sort_index() if len(rel) else pd.Series(dtype=int)
    by_topic = rel["topic"].map(lambda t: names.get(t, str(t))).value_counts() if len(rel) and "topic" in rel.columns else pd.Series(dtype=int)
    ans = Answer("", quotes, counts, by_topic, periods, int(len(rel)), int(len(selected)))
    if len(periods) == 2:
        base = rel if len(rel) >= 10 else selected
        tcol = "topic_name" if "topic_name" in base.columns else "topic"
        ans.compare = compare_periods(base, periods[0], periods[1], topic_col=tcol)
        for p in periods:
            sub = apply_filters(rel, periods=[p]).head(5)
            ans.quotes_by_period[p.label] = [{"date": r["date"].strftime("%d.%m.%Y"), "text": r["clean"],
                                              "topic": names.get(r.get("topic"), "")} for _, r in sub.iterrows()]
    if llm is not None:
        ans.text = _llm_answer(question, ans, llm)
    if not ans.text:
        ans.text = _plain_answer(question, ans, freq)
    return ans


def _plain_answer(question: str, a: Answer, freq: str) -> str:
    per = describe(a.periods)
    lines = [f"Период: {per}. Обращений в выборке: {a.n_selected}, из них похожих на запрос: {a.n_relevant}."]
    if a.counts is not None and len(a.counts):
        fmt = {"D": "%d.%m", "W": "нед. %d.%m", "M": "%m.%Y"}.get(freq, "%d.%m.%Y")
        lines.append("По периодам: " + ", ".join(f"{i.strftime(fmt)}: {int(v)}" for i, v in a.counts.items()))
    if a.by_topic is not None and len(a.by_topic):
        lines.append("По темам: " + ", ".join(f"{t} — {int(v)}" for t, v in a.by_topic.head(6).items()))
    if a.compare is not None and len(a.compare):
        la, lb = a.compare.attrs["labels"]
        up = a.compare[a.compare["delta_share"] > 0].head(3)
        down = a.compare[a.compare["delta_share"] < 0].head(3)
        if len(up):
            lines.append(f"Выросло ({la} → {lb}): " + ", ".join(f"{t} {r['share_a']:.0%} → {r['share_b']:.0%}" for t, r in up.iterrows()))
        if len(down):
            lines.append(f"Снизилось: " + ", ".join(f"{t} {r['share_a']:.0%} → {r['share_b']:.0%}" for t, r in down.iterrows()))
        for label, qs in a.quotes_by_period.items():
            if qs:
                lines.append(f"{label}: " + " | ".join(f"{q['date']}: «{q['text'][:100]}»" for q in qs[:3]))
    elif a.quotes:
        lines.append("Примеры: " + " | ".join(f"{q['date']}: «{q['text'][:120]}»" for q in a.quotes[:5]))
    return "\n".join(lines)


ANSWER_PROMPT = """Ты аналитик обращений абонентов. Отвечай на вопрос пользователя только по приведённым данным, коротко и по делу, по-русски.
Если данных недостаточно — так и скажи. Не выдумывай числа.

Вопрос: «{question}»
Период: {period}. Обращений в выборке: {n_selected}, похожих на запрос: {n_relevant}.
Похожие обращения по периодам: {counts}
Похожие обращения по темам: {by_topic}

Примеры обращений (дата — тема — текст):
{quotes}

Ответ: 3–6 предложений, при необходимости — динамика по периодам и 1–2 цитаты."""


def _llm_answer(question: str, a: Answer, llm) -> str:
    counts = ", ".join(f"{i:%d.%m.%Y}: {int(v)}" for i, v in a.counts.items()) if a.counts is not None and len(a.counts) else "-"
    by_topic = ", ".join(f"{t}: {int(v)}" for t, v in a.by_topic.head(8).items()) if a.by_topic is not None and len(a.by_topic) else "-"
    quotes = "\n".join(f"- {q['date']} — {q['topic']} — {q['text'][:250]}" for q in a.quotes) or "-"
    if a.compare is not None and len(a.compare):
        la, lb = a.compare.attrs["labels"]
        quotes += f"\n\nДоли тем среди похожих обращений, {la} → {lb}:\n" + "\n".join(
            f"- {t}: {r['share_a']:.0%} → {r['share_b']:.0%} ({int(r['count_a'])} → {int(r['count_b'])})" for t, r in a.compare.head(10).iterrows())
        for label, qs in a.quotes_by_period.items():
            quotes += f"\n\nПримеры за {label}:\n" + "\n".join(f"- {q['date']} — {q['text'][:200]}" for q in qs)
    try:
        return llm.chat([{"role": "user", "content": ANSWER_PROMPT.format(
            question=question, period=describe(a.periods), n_selected=a.n_selected, n_relevant=a.n_relevant,
            counts=counts, by_topic=by_topic, quotes=quotes)}]).strip()
    except Exception:
        return ""
