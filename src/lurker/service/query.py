"""Сценарий «вопрос к данным» (ТЗ п. 2, 6, 8): промпт → период + понятие → выборка по смыслу → динамика понятия →
тематики внутри выборки → сравнение периодов → ответ. Работает поверх базового Analysis (очистка, эмбеддинги, обзорные темы).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..cluster import ClusterResult, cluster_embeddings
from .concept import Concept, assign, augment_concept, build_concept, criteria_to_concepts
from ..dynamics import FREQ, Finding, apply_filters, detect_changes
from ..period import Period, describe
from ..pipeline import Analysis
from ..preclass import Criteria, llm_criteria
from ..summarize import TopicSummary, summarize_topics


@dataclass
class QueryResult:
    prompt: str
    concepts: list[Concept]
    periods: list[Period]
    labels: pd.Series                     # метка понятия на каждую строку df (или «прочее»)
    scores: np.ndarray                    # оценка лучшего понятия на строку
    mask: np.ndarray                      # релевантно и попадает в период/фильтры
    in_scope: np.ndarray                  # попадает в период/фильтры
    series: pd.DataFrame                  # индекс — период; count, total, share (+ count_<понятие> при нескольких)
    findings: list[Finding]
    compare: Optional[pd.DataFrame]       # при двух периодах: count/share по понятиям и подтемам
    subtopics: Optional[ClusterResult]
    subsummaries: list[TopicSummary]
    threshold: float
    freq: str
    text: str = ""

    @property
    def n_relevant(self) -> int:
        return int(self.mask.sum())

    @property
    def n_scope(self) -> int:
        return int(self.in_scope.sum())

    def relevant(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.mask]


def run_query(an: Analysis, prompt: str, *, threshold: float = 0.5, freq: Optional[str] = None, llm=None,
              filters: Optional[dict] = None, criteria: Optional[Criteria] = None, min_subtopic_docs: int = 30,
              expand: bool = True, llm_periods_fallback: bool = True, llm_criteria_augment: bool = False) -> QueryResult:
    df = an.df
    freq = freq or an.freq
    ref = df["date"].max().date() if df["date"].notna().any() else date.today()
    nonempty = (df["clean"].str.len() > 3).to_numpy()
    texts = df["clean"].tolist()

    # 1. понятия. Критерии пользователя → несколько понятий (каждая метка — своё). Без критериев — одно понятие из предмета
    # запроса; если есть LLM, её критерии *дополняют* это понятие словами и примерами (у слабых моделей критерии сами по себе
    # хуже промпта: на Qwen3-4B близость к истине 0.20 против 0.61), а лишние метки LLM не участвуют в отборе.
    c, periods = build_concept(prompt, an.embedder, texts, an.E, ref, expand=expand)
    llm_labels: list[str] = []
    if criteria is not None and criteria.labels:
        concepts = criteria_to_concepts(criteria, an.embedder, texts, an.E)
    else:
        concepts = [c] if (c.core or c.proto is not None) else []
        if llm is not None and concepts and llm_criteria_augment:
            # выключено по умолчанию: на Qwen3-4B дополнение критериями роняет точность (0.87 → 0.40 на «домофон в приложении»);
            # включать после проверки на сильной модели (tools/eval_llm.py)
            try:
                crit = llm_criteria(prompt, llm, df.loc[nonempty, "clean"].head(12).tolist())
                llm_labels = [l.name for l in crit.labels]
                augment_concept(c, crit, an.embedder, texts, an.E)
            except Exception:
                pass
    if not periods and llm is not None and llm_periods_fallback:
        # правила надёжнее (на Qwen3-4B совпадение 9/20), LLM — только когда правила не нашли период вовсе
        try:
            from ..period import llm_periods
            periods = llm_periods(prompt, ref, llm, df["date"].min().date(), ref)
        except Exception:
            periods = []

    # 2. отбор
    if concepts:
        labels, S = assign(concepts, texts, an.E, threshold)
        scores = S.max(axis=1) if S.shape[1] else np.zeros(len(df), dtype=np.float32)
        relevant = np.array([l != "прочее" for l in labels]) & nonempty
    else:  # предмета нет («что изменилось за месяц») — вся выборка
        labels, scores, relevant = ["всё"] * len(df), np.ones(len(df), dtype=np.float32), nonempty.copy()
    labels = pd.Series(labels, index=df.index)
    in_scope = np.ones(len(df), dtype=bool)
    if filters or periods:
        in_scope = df.index.isin(apply_filters(df, filters, periods).index)
    mask = relevant & in_scope

    # 3. динамика понятия: count / total / share по периодам (по всей оси времени, не только внутри запрошенного периода)
    base = apply_filters(df, filters)
    g = pd.Grouper(key="date", freq=FREQ.get(freq, freq))
    total = base.groupby(g).size()
    rel_all = base[relevant[base.index]]
    count = rel_all.groupby(g).size().reindex(total.index, fill_value=0)
    series = pd.DataFrame({"count": count, "total": total})
    series["share"] = (series["count"] / series["total"].replace(0, np.nan)).fillna(0)
    if len(concepts) > 1:
        for c in concepts:
            sub = base[(labels[base.index] == c.name).to_numpy()]
            series[f"count_{c.name}"] = sub.groupby(g).size().reindex(total.index, fill_value=0)
    findings = [f for f in detect_changes(series[["count"]].assign(other=series["total"] - series["count"]), spike_factor=2.0)
                if f.topic == "count"]

    # 4. сравнение периодов
    compare = None
    if len(periods) == 2:
        rows = []
        for p in periods:
            sub = apply_filters(base, periods=[p])
            n = len(sub)
            r = int(relevant[sub.index].sum())
            rows.append({"период": p.label, "обращений": n, "релевантных": r, "доля": r / n if n else 0.0})
        compare = pd.DataFrame(rows).set_index("период")

    # 5. тематики внутри выборки
    subtopics, subsummaries = None, []
    idx = np.where(mask)[0]
    if len(idx) >= min_subtopic_docs:
        sub_texts = [texts[i] for i in idx]
        n = len(idx)
        subtopics = cluster_embeddings(an.E[idx], sub_texts, min_cluster_size=max(5, int(round(n ** 0.5 / 2))),
                                       n_neighbors=min(15, max(5, n // 10)))
        for tp in subtopics.topics:
            tp.representatives = [int(idx[i]) for i in tp.representatives]
        sub_df = df.iloc[idx].copy()
        sub_df["topic"] = subtopics.labels
        sub_findings = detect_changes(_timeline(sub_df, "topic", freq))
        subsummaries = summarize_topics(sub_df.reset_index(drop=True), _reindexed(subtopics, idx), sub_findings, topic_col="topic")
        _unique_names(subsummaries, subtopics)
        for s in subsummaries:  # цитаты берём по исходным индексам
            s.quotes = [(df.loc[i, "date"].strftime("%d.%m.%Y") if pd.notna(df.loc[i, "date"]) else "", df.loc[i, "clean"])
                        for i in (subtopics.topic(s.id).representatives[:3] if s.id >= 0 else [])] or s.quotes
        if compare is not None:
            sub_df["date_d"] = sub_df["date"].dt.date
            names = {t.id: t.label for t in subtopics.topics}
            names[-1] = "прочее"
            for p in periods:
                part = sub_df[(sub_df["date_d"] >= p.start) & (sub_df["date_d"] <= p.end)]
                dist = part["topic"].map(names).value_counts()
                compare[f"темы: {p.label}"] = ", ".join(f"{k} — {v}" for k, v in dist.head(5).items())

    qr = QueryResult(prompt, concepts, periods, labels, scores, mask, in_scope, series, findings, compare, subtopics,
                     subsummaries, threshold, freq)
    qr.text = describe_result(qr, df)
    return qr


def _unique_names(summaries, res: ClusterResult) -> None:
    """Имена подтем из ключевых слов могут совпадать — добавляем следующее слово, иначе номер."""
    seen: dict[str, int] = {}
    for s in summaries:
        name = s.name
        if name in seen:
            tp = res.topic(s.id)
            extra = [k for k in (tp.keywords if tp else []) if k not in name][:1]
            name = f"{s.name}, {extra[0]}" if extra else f"{s.name} ({seen[s.name] + 1})"
        seen[s.name] = seen.get(s.name, 0) + 1
        seen.setdefault(name, 0)
        s.name = name
        tp = res.topic(s.id)
        if tp:
            tp.name = name


def _timeline(sub_df: pd.DataFrame, col: str, freq: str) -> pd.DataFrame:
    d = sub_df[sub_df["date"].notna()]
    if d.empty:
        return pd.DataFrame()
    return d.groupby([pd.Grouper(key="date", freq=FREQ.get(freq, freq)), col]).size().unstack(fill_value=0).sort_index()


def _reindexed(res: ClusterResult, idx: np.ndarray) -> ClusterResult:
    """Копия результата с представителями в координатах подвыборки (для summarize_topics)."""
    import copy
    pos = {int(g): k for k, g in enumerate(idx)}
    r = copy.copy(res)
    r.topics = [copy.copy(t) for t in res.topics]
    for t in r.topics:
        t.representatives = [pos[i] for i in t.representatives if i in pos]
    return r


def describe_result(q: QueryResult, df: pd.DataFrame) -> str:
    lines = []
    what = " / ".join(c.name for c in q.concepts) or "все обращения"
    lines.append(f"Понятие: {what}. Период: {describe(q.periods)}. Релевантных обращений: {q.n_relevant} из {q.n_scope} "
                 f"({q.n_relevant / q.n_scope:.1%})." if q.n_scope else "Нет обращений в выбранном периоде.")
    if q.concepts and q.concepts[0].keywords:
        lines.append("Ключевые слова: " + ", ".join(q.concepts[0].keywords[:12]) + ".")
    if q.findings:
        lines.append("Динамика: " + "; ".join(f.describe(what) for f in q.findings[:4]) + ".")
    elif len(q.series) >= 3:
        lines.append("Динамика: без заметных изменений.")
    if q.compare is not None:
        a, b = q.compare.index[:2]
        ra, rb = q.compare.loc[a], q.compare.loc[b]
        lines.append(f"Сравнение: {a} — {int(ra['релевантных'])} из {int(ra['обращений'])} ({ra['доля']:.1%}), "
                     f"{b} — {int(rb['релевантных'])} из {int(rb['обращений'])} ({rb['доля']:.1%}).")
    if q.subsummaries:
        tops = [s for s in q.subsummaries if s.id >= 0][:6]
        lines.append("Тематики внутри: " + "; ".join(f"{s.name} ({s.count})" for s in tops) + ".")
    return "\n".join(lines)


def borderline(q: QueryResult, df: pd.DataFrame, width: float = 0.06, n: int = 8) -> pd.DataFrame:
    """Обращения у порога — для калибровки ползунка."""
    s = pd.Series(q.scores, index=df.index)
    near = df[(s >= q.threshold - width) & (s <= q.threshold + width) & q.in_scope].assign(score=s)
    return near.sort_values("score", ascending=False)[["date", "clean", "score"]].head(n)
