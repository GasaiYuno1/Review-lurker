"""Динамика по времени без LLM (ТЗ п. 6.1): счётчики «тема × период» с гранулярностью день/неделя/месяц,
фильтры по любым колонкам, сравнение периодов и автоматический поиск всплесков/новых тем/трендов.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .period import Period

FREQ = {"D": "D", "W": "W-MON", "M": "MS", "day": "D", "week": "W-MON", "month": "MS"}


def apply_filters(df: pd.DataFrame, filters: Optional[dict] = None, periods: Optional[Sequence[Period]] = None) -> pd.DataFrame:
    out = df
    for col, val in (filters or {}).items():
        if val is None or val == "" or col not in out.columns:
            continue
        vals = val if isinstance(val, (list, tuple, set)) else [val]
        out = out[out[col].isin(vals)]
    if periods:
        d = out["date"].dt.date
        mask = np.zeros(len(out), dtype=bool)
        for p in periods:
            mask |= ((d >= p.start) & (d <= p.end)).to_numpy()
        out = out[mask]
    return out


def timeline(df: pd.DataFrame, topic_col: str = "topic", freq: str = "W", filters: Optional[dict] = None,
             shares: bool = False, min_total: int = 1) -> pd.DataFrame:
    """Таблица: индекс — начало периода, колонки — темы, значения — число (или доля) комментариев."""
    d = apply_filters(df, filters)
    d = d[d["date"].notna()]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby([pd.Grouper(key="date", freq=FREQ.get(freq, freq)), topic_col]).size().unstack(fill_value=0)
    g = g.sort_index()
    g = g.loc[:, g.sum() >= min_total]
    if shares:
        g = g.div(g.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return g


def totals(df: pd.DataFrame, freq: str = "W", filters: Optional[dict] = None) -> pd.Series:
    d = apply_filters(df, filters)
    d = d[d["date"].notna()]
    return d.groupby(pd.Grouper(key="date", freq=FREQ.get(freq, freq))).size().sort_index()


def compare_periods(df: pd.DataFrame, a: Period, b: Period, topic_col: str = "topic", filters: Optional[dict] = None) -> pd.DataFrame:
    """Что изменилось между периодами: число и доля по темам, сдвиг доли и отношение (с добавкой +1 против деления на 0)."""
    d = apply_filters(df, filters)
    ca = apply_filters(d, periods=[a])[topic_col].value_counts()
    cb = apply_filters(d, periods=[b])[topic_col].value_counts()
    idx = ca.index.union(cb.index)
    ca, cb = ca.reindex(idx, fill_value=0), cb.reindex(idx, fill_value=0)
    ta, tb = max(ca.sum(), 1), max(cb.sum(), 1)
    out = pd.DataFrame({"count_a": ca, "count_b": cb, "share_a": ca / ta, "share_b": cb / tb})
    out["delta_share"] = out["share_b"] - out["share_a"]
    out["ratio"] = (out["share_b"] + 1 / tb) / (out["share_a"] + 1 / ta)
    out.attrs["labels"] = (a.label, b.label)
    return out.sort_values("delta_share", key=np.abs, ascending=False)


@dataclass
class Finding:
    kind: str          # spike | new | growth | decline
    topic: object
    period: Optional[pd.Timestamp]
    count: float
    baseline: float
    factor: float

    def describe(self, name: Optional[str] = None) -> str:
        t = name or str(self.topic)
        if self.kind == "spike":
            return f"«{t}»: всплеск {self.period:%d.%m.%Y} — {int(self.count)} против обычных {self.baseline:.0f} (×{self.factor:.1f})"
        if self.kind == "new":
            return f"«{t}»: новая тема, появилась {self.period:%d.%m.%Y} ({int(self.count)} за период)"
        if self.kind == "growth":
            return f"«{t}»: рост — последняя треть периода {self.count:.0f} обращений против {self.baseline:.0f} в первой (×{self.factor:.1f})"
        return f"«{t}»: спад — последняя треть периода {self.count:.0f} обращений против {self.baseline:.0f} в первой (×{self.factor:.2f})"


def detect_changes(tl: pd.DataFrame, min_count: int = 6, spike_factor: float = 2.5, spike_z: float = 3.0,
                   trend_factor: float = 1.8, max_findings: int = 20) -> list[Finding]:
    """На входе таблица счётчиков (timeline без shares). Три типа находок:
    spike — период, где счётчик ≥ spike_factor × медиана остальных периодов и отклонение ≥ spike_z·(√медиана + 1);
    new — тема с нулями в первой трети таблицы и заметным объёмом после появления;
    growth/decline — число обращений в последней трети периода против первой (×trend_factor).
    Неполные крайние периоды (суммарный объём < половины медианного) в тренде не участвуют."""
    findings: list[Finding] = []
    if tl.empty or len(tl) < 3:
        return findings
    tot = tl.sum(axis=1).to_numpy(dtype=float)
    med = float(np.median(tot))
    lo = 0 if tot[0] >= 0.5 * med else 1
    hi = len(tl) if tot[-1] >= 0.5 * med else len(tl) - 1
    n = len(tl)
    for topic in tl.columns:
        s = tl[topic].astype(float)
        vals = s.to_numpy()
        nz = np.nonzero(vals)[0]
        if len(nz) and nz[0] >= max(2, n // 3) and vals[nz[0]:].sum() >= min_count:
            findings.append(Finding("new", topic, s.index[nz[0]], vals[nz[0]:].sum(), 0.0, float("inf")))
            continue
        for i in range(n):
            others = np.delete(vals, i)
            base = float(np.median(others))
            v = vals[i]
            if v >= min_count and v >= spike_factor * max(base, 1.0) and (v - base) / (np.sqrt(max(base, 1.0)) + 1) >= spike_z:
                findings.append(Finding("spike", topic, s.index[i], v, base, v / max(base, 1.0)))
        core = vals[lo:hi]
        k = max(1, len(core) // 3)
        if len(core) >= 3:
            first, last = core[:k].sum(), core[-k:].sum()
            if first + last >= 2 * min_count:
                ratio = (last + 1) / (first + 1)
                if ratio >= trend_factor:
                    findings.append(Finding("growth", topic, None, last, first, ratio))
                elif ratio <= 1 / trend_factor:
                    findings.append(Finding("decline", topic, None, last, first, ratio))
    order = {"spike": 0, "new": 1, "growth": 2, "decline": 3}
    findings.sort(key=lambda f: (order[f.kind], -abs(np.log(f.factor) if np.isfinite(f.factor) else 10)))
    return findings[:max_findings]
