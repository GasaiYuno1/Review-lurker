"""Новизна комментариев (ТЗ v2, задача 1: «какие необычные проблемы встречаются», «какие темы появляются»).

unusual_scores — насколько комментарий непохож на остальные: 1 − среднее сходство с k ближайшими соседями по всему
  корпусу, робастный z. Короткие тексты (< min_words слов: опечатки, «см. предыдущий») не ранжируются — необычной
  проблемы они не описывают. Проверка (research/comments/novelty.py): подмешанные необычные обращения находятся с AP 0.86,
  контрольные обычные того же автора — нет; на настоящих отзывах — рабочий «список на прочтение», а не фильтр.
emerging_groups — новые темы по периодам: тексты периода сравниваются со всеми текстами до него; «новые» — дальше от
  прошлого, чем `q`-й процентиль той же величины внутри прошлого; новые тексты группируются (агломеративная, косинус),
  группа ≥ min_group, у которой и центр далёк от прошлого (выше centroid_q-го процентиля), — кандидат в новую тему. На синтетических тикетах заложенная
  новая тема найдена в первый же месяц, ложных кандидатов в остальные месяцы — ни одного.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .anomaly import knn_score, robust_z


def unusual_scores(E: np.ndarray, texts: Sequence[str], k: int = 10, min_words: int = 4) -> np.ndarray:
    """z-оценка необычности каждого текста (−inf для коротких)."""
    one = np.zeros(len(E), dtype=int)
    z = robust_z(knn_score(np.asarray(E, dtype=np.float32), one, k=k), one)
    z[np.array([len(str(t).split()) < min_words for t in texts])] = -np.inf
    return z


def _knn_novelty(Q: np.ndarray, R: np.ndarray, k: int, exclude_self: bool = False, block: int = 2048) -> np.ndarray:
    out = np.zeros(len(Q))
    for a in range(0, len(Q), block):
        S = Q[a:a + block] @ R.T
        if exclude_self:
            S[np.arange(S.shape[0]), np.arange(a, a + S.shape[0])] = -np.inf
        kk = min(k, S.shape[1] - (1 if exclude_self else 0))
        out[a:a + block] = 1 - np.sort(S, axis=1)[:, -kk:].mean(axis=1)
    return out


def novel_by_period(E: np.ndarray, dates: Sequence, freq: str = "M", k: int = 5, q: float = 99, min_ref: int = 200,
                    with_ref: bool = False):
    """Для каждого периода → (период, индексы текстов периода, индексы «новых» среди них[, эталон, порог])."""
    E = np.asarray(E, dtype=np.float32)
    per = pd.to_datetime(pd.Series(dates)).dt.to_period(freq).to_numpy()
    for m in sorted(set(p for p in per if not pd.isna(p))):
        ref, cur = np.where(per < m)[0], np.where(per == m)[0]
        if len(ref) < min_ref or not len(cur):
            continue
        nov = _knn_novelty(E[cur], E[ref], k)
        thr = np.percentile(_knn_novelty(E[ref], E[ref], k, exclude_self=True), q)
        yield (m, cur, cur[nov > thr], ref, thr) if with_ref else (m, cur, cur[nov > thr])


def emerging_groups(E: np.ndarray, dates: Sequence, freq: str = "M", k: int = 5, q: float = 99, min_group: int = 5,
                    dist: float = 0.35, min_ref: int = 200, centroid_q: float = 90) -> pd.DataFrame:
    """→ таблица кандидатов в новые темы: period, size, members (индексы текстов). Периоды, у которых в прошлом меньше
    min_ref текстов, пропускаются (не с чем сравнивать)."""
    from sklearn.cluster import AgglomerativeClustering
    E = np.asarray(E, dtype=np.float32)
    rows = []
    for m, _cur, new, ref, _thr in novel_by_period(E, dates, freq, k, q, min_ref, with_ref=True):
        if len(new) < min_group:
            continue
        thr_c = np.percentile(_knn_novelty(E[ref], E[ref], k, exclude_self=True), centroid_q)
        g = AgglomerativeClustering(n_clusters=None, distance_threshold=dist, metric="cosine", linkage="average").fit_predict(E[new])
        for gid in np.unique(g):
            idx = new[g == gid]
            if len(idx) < min_group:
                continue
            c = E[idx].mean(axis=0)
            # центр группы тоже должен быть далёк от прошлого: у группы пограничных текстов старой темы шум усредняется,
            # и центр ложится в самую гущу старой темы (≈ 0-й процентиль новизны), у настоящей новой темы — 98–99-й
            if _knn_novelty((c / np.linalg.norm(c))[None, :], E[ref], k)[0] <= thr_c:
                continue
            rows.append({"period": str(m), "size": int(len(idx)), "members": idx.tolist()})
    return pd.DataFrame(rows, columns=["period", "size", "members"])
