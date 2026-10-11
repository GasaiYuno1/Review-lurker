"""Поиск аномалий среди текстов одной темы (ТЗ v2, задача 2: «аномальные звонки»).

Схема из ТЗ: кластеризуем все звонки по темам и только внутри темы ищем звонки, сильно отличающиеся от остальных.
Оценки аномальности (больше — подозрительнее), все нормируются внутри темы робастным z (медиана, MAD), чтобы темы
с разным разбросом были сравнимы и порог был один:
  centroid — косинусное расстояние до центра темы (базовая схема ТЗ);
  knn      — среднее расстояние до k ближайших звонков той же темы (локальная плотность: ловит звонок,
             у которого нет похожих, даже если он недалеко от центра);
  lof      — Local Outlier Factor внутри темы;
  turns    — по репликам: каждая реплика оператора сравнивается с k ближайшими репликами операторов в других
             звонках той же темы; оценка звонка — максимум по репликам. Ловит одну вставленную фразу
             (грубость, бред), которая при усреднении всего диалога в один вектор растворяется;
  coherence — связность: начало, середина и конец звонка должны быть об одном (ловит подмену куска диалога);
  relevance — уместность: реплика оператора против предыдущей реплики клиента (ловит ответы невпопад).
Метрики: precision@K, recall@K, average precision, ROC-AUC; доля «испорченных» (шум STT) в топ-K — ложные тревоги.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np


OP_LABELS = ("оператор", "менеджер", "сотрудник", "специалист", "агент", "operator", "agent")
CLIENT_LABELS = ("клиент", "абонент", "пользователь", "покупатель", "client", "customer", "user")


def split_dialog(text: str, op_labels: Sequence[str] = OP_LABELS, client_labels: Sequence[str] = CLIENT_LABELS) -> list[tuple[str, str]]:
    """Расшифровка с метками говорящих («Оператор: …», «Клиент: …», в любом месте текста, с переносами строк и без) →
    [(\"op\" | \"cl\", реплика)]. Метки — без учёта регистра; для STT вида «Спикер 1:» передать их явно."""
    labels = {l.lower(): "op" for l in op_labels} | {l.lower(): "cl" for l in client_labels}
    alt = "|".join(re.escape(l) for l in sorted(labels, key=len, reverse=True))
    rx = re.compile(rf"(?:^|(?<=[\s.!?…»)]))({alt})\s*:\s*", re.IGNORECASE)
    text = str(text or "").strip()
    marks = list(rx.finditer(text))
    out = []
    for i, m in enumerate(marks):
        part = " ".join(text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)].split())
        if part:
            out.append((labels[m.group(1).lower()], part))
    return out


def center(E: np.ndarray) -> np.ndarray:
    """Вычесть средний вектор корпуса и нормировать заново. Эмбеддинги однотипных текстов (реплики звонков в поддержку)
    сильно сонаправлены — общий «жанровый» компонент забивает смысловые различия; центрирование их проявляет."""
    E = np.asarray(E, dtype=np.float32) - np.asarray(E, dtype=np.float32).mean(axis=0)
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)


def assign_topics(E: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Все выбросы HDBSCAN (-1) — к ближайшему центроиду: в каждой теме должен быть каждый звонок."""
    labels = np.asarray(labels).copy()
    ids = sorted(set(labels[labels >= 0].tolist()))
    if not ids:
        return np.zeros(len(E), dtype=int)
    C = np.stack([_unit(E[labels == i].mean(axis=0)) for i in ids])
    out = labels < 0
    if out.any():
        labels[out] = np.array(ids)[np.argmax(E[out] @ C.T, axis=1)]
    return labels


def robust_z(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """(x − медиана группы) / (1.4826·MAD группы); маленькие группы (< 5) — по общей медиане/MAD."""
    x = np.asarray(x, dtype=np.float64)
    z = np.zeros_like(x)
    gmed, gmad = np.median(x), _mad(x)
    for g in np.unique(groups):
        m = groups == g
        med, mad = (np.median(x[m]), _mad(x[m])) if m.sum() >= 5 else (gmed, gmad)
        z[m] = (x[m] - med) / (mad or gmad or 1.0)
    return z


def centroid_score(E: np.ndarray, topics: np.ndarray) -> np.ndarray:
    """Расстояние до центра своей темы, центр — без самого звонка (leave-one-out)."""
    s = np.zeros(len(E))
    for t in np.unique(topics):
        idx = np.where(topics == t)[0]
        tot = E[idx].sum(axis=0)
        for i in idx:
            c = tot - E[i] if len(idx) > 1 else tot
            s[i] = 1.0 - float(E[i] @ _unit(c))
    return s


def knn_score(E: np.ndarray, topics: np.ndarray, k: int = 10) -> np.ndarray:
    s = np.zeros(len(E))
    for t in np.unique(topics):
        idx = np.where(topics == t)[0]
        if len(idx) < 2:
            continue
        S = E[idx] @ E[idx].T
        np.fill_diagonal(S, -np.inf)
        kk = min(k, len(idx) - 1)
        top = -np.sort(-S, axis=1)[:, :kk]
        s[idx] = 1.0 - top.mean(axis=1)
    return s


def lof_score(E: np.ndarray, topics: np.ndarray, k: int = 15) -> np.ndarray:
    from sklearn.neighbors import LocalOutlierFactor
    s = np.zeros(len(E))
    for t in np.unique(topics):
        idx = np.where(topics == t)[0]
        if len(idx) < 5:
            continue
        lof = LocalOutlierFactor(n_neighbors=min(k, len(idx) - 1), metric="cosine")
        lof.fit(E[idx])
        s[idx] = -lof.negative_outlier_factor_
    return s


def turn_novelty(T: np.ndarray, turn_doc: np.ndarray, doc_topics: np.ndarray, k: int = 10, block: int = 2048) -> np.ndarray:
    """Новизна каждой реплики: 1 − среднее сходство с k ближайшими репликами из ДРУГИХ звонков той же темы.
    Считается блоками по `block` реплик — память O(block × размер темы), а не квадрат темы."""
    T = np.asarray(T, dtype=np.float32)
    turn_doc = np.asarray(turn_doc)
    turn_topic = np.asarray(doc_topics)[turn_doc]
    ts = np.zeros(len(T))
    for t in np.unique(turn_topic):
        idx = np.where(turn_topic == t)[0]
        if len(idx) < 3:
            continue
        kk = max(1, min(k, len(idx) - 1))
        Tt, dt = T[idx], turn_doc[idx]
        for a in range(0, len(idx), block):
            S = Tt[a:a + block] @ Tt.T
            S[dt[a:a + block, None] == dt[None, :]] = -np.inf
            best = -np.sort(-S, axis=1)[:, :kk]
            best[~np.isfinite(best)] = np.nan
            with np.errstate(all="ignore"):
                ts[idx[a:a + block]] = 1.0 - np.nan_to_num(np.nanmean(best, axis=1), nan=0.0)
    return ts


def turn_score(T: np.ndarray, turn_doc: np.ndarray, doc_topics: np.ndarray, k: int = 10, top: int = 1) -> np.ndarray:
    """T — эмбеддинги реплик оператора, turn_doc — номер звонка для каждой реплики.
    Реплика — `turn_novelty`; звонок — среднее из `top` наибольших оценок его реплик (top=1 — максимум)."""
    return doc_max(turn_novelty(T, turn_doc, doc_topics, k=k), turn_doc, len(doc_topics), top=top)


def doc_max(ts: np.ndarray, turn_doc: np.ndarray, n_docs: int, top: int = 1) -> np.ndarray:
    """Оценка звонка по его репликам: среднее из `top` наибольших (звонок без реплик — 0)."""
    ts, turn_doc = np.asarray(ts, dtype=np.float64), np.asarray(turn_doc)
    out = np.zeros(n_docs)
    if not len(ts):
        return out
    order = np.lexsort((-ts, turn_doc))
    d, v = turn_doc[order], ts[order]
    starts = np.r_[0, np.flatnonzero(np.diff(d)) + 1]
    for a, b in zip(starts, np.r_[starts[1:], len(d)]):
        out[d[a]] = v[a:min(b, a + top)].mean()
    return out


def offtopic_score(T: np.ndarray, turn_doc: np.ndarray, doc_topics: np.ndarray, top: int = 3) -> np.ndarray:
    """«Чужая тема по репликам» — для подмешанного куска чужого разговора: центр каждой темы считается по репликам её
    звонков; запас реплики = сходство с ближайшей ЧУЖОЙ темой − сходство со своей (> 0 — реплика ближе к другой теме);
    звонок — среднее из `top` наибольших запасов. T лучше центрировать (center)."""
    T = np.asarray(T, dtype=np.float32)
    turn_doc = np.asarray(turn_doc)
    tt = np.asarray(doc_topics)[turn_doc]
    ids = np.unique(tt)
    C = np.stack([_unit(T[tt == t].mean(axis=0)) for t in ids])
    S = T @ C.T
    pos = np.searchsorted(ids, tt)
    own = S[np.arange(len(T)), pos]
    S[np.arange(len(T)), pos] = -np.inf
    return doc_max(S.max(axis=1) - own, turn_doc, len(doc_topics), top=top)


def coherence_score(T: np.ndarray, turn_doc: np.ndarray, n_docs: int, parts: int = 3) -> np.ndarray:
    """Связность звонка: реплики делятся на `parts` частей по порядку, часть = средний вектор её реплик;
    оценка = 1 − минимальное сходство соседних частей. Подмена середины диалога чужой темой ломает связность,
    даже если звонок в целом похож на свою тему. T — эмбеддинги ВСЕХ реплик в порядке диалога."""
    turn_doc = np.asarray(turn_doc)
    out = np.zeros(n_docs)
    for d in range(n_docs):
        idx = np.where(turn_doc == d)[0]
        if len(idx) < parts:
            continue
        chunks = np.array_split(idx, parts)
        V = np.stack([_unit(T[c].mean(axis=0)) for c in chunks])
        out[d] = 1.0 - min(float(V[i] @ V[i + 1]) for i in range(parts - 1))
    return out


def relevance_score(T: np.ndarray, turn_doc: np.ndarray, speaker: Sequence[str], n_docs: int, top: int = 1) -> np.ndarray:
    """Уместность ответов оператора: для каждой реплики оператора — 1 − сходство с предыдущей репликой клиента
    в том же звонке; звонок — среднее из `top` худших. Грубость и бред обычно не отвечают на вопрос клиента."""
    turn_doc = np.asarray(turn_doc)
    speaker = np.asarray(speaker)
    per_doc: dict[int, list[float]] = {}
    for i in range(1, len(T)):
        if speaker[i] == "op" and speaker[i - 1] == "cl" and turn_doc[i] == turn_doc[i - 1]:
            per_doc.setdefault(int(turn_doc[i]), []).append(1.0 - float(T[i] @ T[i - 1]))
    out = np.zeros(n_docs)
    for d, v in per_doc.items():
        out[d] = float(np.mean(sorted(v, reverse=True)[:top]))
    return out


# ---------- метрики ----------

def precision_at_k(scores: np.ndarray, positive: np.ndarray, k: int) -> float:
    order = np.argsort(-np.asarray(scores))[:k]
    return float(np.asarray(positive)[order].mean()) if k else 0.0


def recall_at_k(scores: np.ndarray, positive: np.ndarray, k: int) -> float:
    positive = np.asarray(positive)
    order = np.argsort(-np.asarray(scores))[:k]
    return float(positive[order].sum() / max(positive.sum(), 1))


def share_at_k(scores: np.ndarray, flag: np.ndarray, k: int) -> float:
    return precision_at_k(scores, flag, k)


def ranking_metrics(scores: np.ndarray, positive: np.ndarray, ks: Sequence[int] = (25, 50, 100, 150),
                    corrupted: Optional[np.ndarray] = None) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    positive = np.asarray(positive).astype(bool)
    out = {"ap": float(average_precision_score(positive, scores)) if positive.any() else 0.0,
           "auc": float(roc_auc_score(positive, scores)) if 0 < positive.sum() < len(positive) else 0.5}
    for k in ks:
        out[f"p@{k}"] = precision_at_k(scores, positive, k)
        out[f"r@{k}"] = recall_at_k(scores, positive, k)
        if corrupted is not None:
            out[f"noise@{k}"] = share_at_k(scores, np.asarray(corrupted).astype(bool), k)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def _mad(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))
