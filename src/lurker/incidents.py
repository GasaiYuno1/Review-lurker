"""Сбои по дням без разметки событий: всплески тем, слова, которые их объясняют, и сценарий аварии в словах оператора.

Проверено на месяце звонков заказчика (42 600): 18 всплесков тем в 2–9 раз, слова дня сразу говорят, что случилось,
часть всплесков подтверждается ростом той же темы в соцсетях (research/REPORT.md, прод-данные).

- daily_spikes(counts) — день × тема: всплеск, когда темы в разы больше ожидаемого по её доле за период и объёму дня
  (пуассоновский хвост с поправкой Бонферрони на число клеток).
- spike_words(texts, days, topics, spikes) — слова, которые в текстах темы в день всплеска встречаются заметно чаще,
  чем в той же теме в остальные дни (лог-отношение долей текстов со словом, сглаживание +1).
- outage_days(texts, days) — доля звонков дня со сценарием аварии («ведутся работы», «приносим извинения за
  неудобства», «авария», «массовый сбой») и её отношение к медиане по дням.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson

WORD = re.compile(r"[а-яё]{4,}")
STOP = set("""это было была были если когда даже чтобы только тоже потом сейчас здесь тогда этот того этого этой этом
такой какой какая много можно нельзя будет было есть всего всех всем вообще вобще просто значит смотрите скажите
подскажите хорошо понял поняла минуту секунду пожалуйста спасибо давайте сейчас ладно вот так""".split())
OUTAGE = re.compile(r"ведутся\s+(?:\w+\s+){0,2}работы|восстановительн\w*\s+работ|авари\w*|"
                    r"прино\w*\s+(?:свои\s+)?извинени\w*\s+за\s+(?:\w+\s+)?неудобств|массов\w*\s+(?:сбо|проблем|неисправн)")


def daily_counts(days: Sequence, topics: Sequence) -> pd.DataFrame:
    """Таблица день × тема (число текстов)."""
    return pd.DataFrame({"day": list(days), "topic": list(topics)}).groupby(["day", "topic"]).size().unstack(fill_value=0)


def daily_spikes(counts: pd.DataFrame, min_n: int = 15, min_ratio: float = 2.0, alpha: float = 0.01) -> pd.DataFrame:
    """Всплески: n ≥ min_n, n ≥ min_ratio × ожидаемого и пуассоновский p < alpha / (дней × тем).
    Ожидаемое — объём дня × доля темы за весь период."""
    t = counts.astype(float)
    tot = t.sum(axis=1).values
    share = t.sum(axis=0).values / max(tot.sum(), 1)
    exp = np.outer(tot, share)
    p = poisson.sf(t.values - 1, np.maximum(exp, 1e-9))
    m = (t.values >= min_n) & (t.values >= min_ratio * exp) & (p < alpha / max(t.size, 1))
    rows = [{"day": t.index[i], "topic": t.columns[j], "n": int(t.values[i, j]), "expected": round(float(exp[i, j]), 1),
             "ratio": round(float(t.values[i, j] / max(exp[i, j], 1e-9)), 1), "p": float(p[i, j])}
            for i, j in zip(*np.where(m))]
    out = pd.DataFrame(rows, columns=["day", "topic", "n", "expected", "ratio", "p"])
    return out.sort_values(["day", "ratio"], ascending=[True, False]).reset_index(drop=True)


def spike_words(texts: Sequence[str], days: Sequence, topics: Sequence, spikes: pd.DataFrame,
                top: int = 12, min_count: int = 5, min_share: float = 0.15) -> pd.DataFrame:
    """Для каждого всплеска — слова дня: доля текстов темы со словом в этот день против остальных дней."""
    days, topics = np.asarray(list(days), dtype=object), np.asarray(list(topics), dtype=object)
    words = [set(WORD.findall(str(x).lower())) - STOP for x in texts]
    rows = []
    for r in spikes.itertuples():
        same = np.where(topics == r.topic)[0]
        on = [i for i in same if days[i] == r.day]
        off = [i for i in same if days[i] != r.day]
        c_on, c_off = Counter(w for i in on for w in words[i]), Counter(w for i in off for w in words[i])
        sc = {w: np.log((c_on[w] + 1) / (len(on) + 2)) - np.log((c_off[w] + 1) / (len(off) + 2))
              for w in c_on if c_on[w] >= max(min_count, min_share * len(on))}
        best = sorted(sc, key=sc.get, reverse=True)[:top]
        rows.append({"day": r.day, "topic": r.topic, "n": len(on), "ratio": r.ratio,
                     "words": ", ".join(f"{w} {c_on[w] / max(len(on), 1):.0%}/{c_off[w] / max(len(off), 1):.0%}"
                                        for w in best)})
    return pd.DataFrame(rows, columns=["day", "topic", "n", "ratio", "words"])


def outage_days(texts: Sequence[str], days: Sequence) -> pd.DataFrame:
    """День → звонков, со сценарием аварии, доля, раз к медиане доли по дням."""
    hit = [bool(OUTAGE.search(str(x).lower())) for x in texts]
    d = pd.DataFrame({"day": list(days), "outage": hit}).groupby("day").agg(calls=("outage", "size"),
                                                                             outage=("outage", "sum"))
    d["share"] = (d.outage / d.calls).round(4)
    med = d.share.median()
    d["ratio_to_median"] = (d.share / med).round(1) if med > 0 else np.nan
    return d
