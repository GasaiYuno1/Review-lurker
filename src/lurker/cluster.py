"""Кластеризация очищенных текстов по эмбеддингам.

Основной путь — схема BERTopic: UMAP → HDBSCAN → c-TF-IDF, с настройкой под русский и небольшие датасеты
(`method="umap_hdbscan"`, без зависимости от самой библиотеки; `method="bertopic"` — то же через библиотеку, если она есть).
Запасной путь — агломеративная кластеризация по косинусу (`method="agglomerative"`, без umap/hdbscan).
Выбросы HDBSCAN (-1) дораспределяются по ближайшему центроиду при сходстве ≥ outlier_threshold, остальное — «прочее» (-1).
Ключевые слова кластера — c-TF-IDF по леммам и биграммам, представители — ближайшие к центроиду тексты.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import text as T


@dataclass
class TopicInfo:
    id: int
    size: int
    keywords: list[str]
    representatives: list[int]          # индексы документов
    name: str = ""
    centroid: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        words = [k for k in self.keywords if " " not in k][:3] or self.keywords[:2]
        return ", ".join(words) or f"тема {self.id}"


@dataclass
class ClusterResult:
    labels: np.ndarray                  # -1 = прочее
    topics: list[TopicInfo]
    method: str
    params: dict = field(default_factory=dict)

    @property
    def n_outliers(self) -> int:
        return int((self.labels == -1).sum())

    def topic(self, tid: int) -> Optional[TopicInfo]:
        return next((t for t in self.topics if t.id == tid), None)


def default_min_cluster_size(n: int) -> int:
    return max(8, int(round(n ** 0.5 / 2)))


def cluster_embeddings(E: np.ndarray, texts: Sequence[str], *, method: str = "umap_hdbscan",
                       min_cluster_size: Optional[int] = None, min_samples: Optional[int] = None,
                       n_neighbors: int = 15, n_components: int = 5, outlier_threshold: float = 0.35,
                       distance_threshold: float = 0.55, random_state: int = 42,
                       max_topics: Optional[int] = None, merge_threshold: Optional[float] = None,
                       n_keywords: int = 8, n_representatives: int = 5) -> ClusterResult:
    n = len(texts)
    E = np.asarray(E, dtype=np.float32)
    mcs = min_cluster_size or default_min_cluster_size(n)
    ms = min_samples or max(3, mcs // 3)
    params = {"method": method, "min_cluster_size": mcs, "min_samples": ms, "n_neighbors": n_neighbors,
              "n_components": n_components, "outlier_threshold": outlier_threshold, "max_topics": max_topics,
              "merge_threshold": merge_threshold}

    if n < 2 * mcs:
        labels = np.zeros(n, dtype=int) if n else np.array([], dtype=int)
        method = "single"
    elif method == "bertopic":
        labels = _bertopic(E, texts, mcs, ms, n_neighbors, n_components, random_state)
    elif method == "umap_hdbscan":
        labels = _umap_hdbscan(E, mcs, ms, n_neighbors, n_components, random_state)
    elif method == "agglomerative":
        labels = _agglomerative(E, mcs, distance_threshold)
        params["distance_threshold"] = distance_threshold
    else:
        raise ValueError(f"неизвестный метод кластеризации: {method}")

    labels = _relabel(labels)
    labels = reduce_topics(E, labels, max_topics=max_topics, merge_threshold=merge_threshold)
    labels = _reassign_outliers(E, labels, outlier_threshold)
    topics = describe_clusters(E, texts, labels, n_keywords=n_keywords, n_representatives=n_representatives)
    return ClusterResult(labels, topics, method, params)


# ---------- методы ----------

def _umap_hdbscan(E, mcs, ms, n_neighbors, n_components, random_state):
    import hdbscan
    import umap
    n = len(E)
    nn = max(2, min(n_neighbors, n - 1))
    nc = max(2, min(n_components, n - 2))
    red = umap.UMAP(n_neighbors=nn, n_components=nc, min_dist=0.0, metric="cosine", random_state=random_state)
    Z = red.fit_transform(E)
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean", cluster_selection_method="eom")
    return cl.fit_predict(Z)


def _bertopic(E, texts, mcs, ms, n_neighbors, n_components, random_state):
    import hdbscan
    import umap
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    n = len(E)
    model = BERTopic(
        embedding_model=None, language="multilingual",
        umap_model=umap.UMAP(n_neighbors=max(2, min(n_neighbors, n - 1)), n_components=max(2, min(n_components, n - 2)),
                             min_dist=0.0, metric="cosine", random_state=random_state),
        hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                                      cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(analyzer=T.analyzer, min_df=2),
        calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(list(texts), embeddings=E)
    return np.asarray(topics)


def _agglomerative(E, mcs, distance_threshold):
    from sklearn.cluster import AgglomerativeClustering
    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
    labels = cl.fit_predict(E)
    sizes = np.bincount(labels)
    small = sizes[labels] < mcs
    labels = labels.copy()
    labels[small] = -1
    return labels


# ---------- постобработка ----------

def _relabel(labels: np.ndarray) -> np.ndarray:
    """Кластеры по убыванию размера: 0 — самый большой; -1 остаётся."""
    labels = np.asarray(labels)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping.get(l, -1) for l in labels], dtype=int)


def reduce_topics(E: np.ndarray, labels: np.ndarray, max_topics: Optional[int] = None,
                  merge_threshold: Optional[float] = None) -> np.ndarray:
    """Слияние близких кластеров по центроидам (агломеративно, среднее косинусное расстояние):
    до max_topics штук и/или пока сходство центроидов ≥ merge_threshold. Как reduce_topics в BERTopic."""
    cents = centroids(E, labels)
    ids = sorted(cents)
    if len(ids) < 2 or (not max_topics and not merge_threshold):
        return labels
    if max_topics and len(ids) <= max_topics and not merge_threshold:
        return labels
    from sklearn.cluster import AgglomerativeClustering
    C = np.stack([cents[i] for i in ids])
    kw = {"metric": "cosine", "linkage": "average"}
    if max_topics and len(ids) > max_topics:
        groups = AgglomerativeClustering(n_clusters=max_topics, **kw).fit_predict(C)
        if merge_threshold:
            # после укрупнения до max_topics ещё доклеиваем очень близкие
            g2 = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - merge_threshold, **kw).fit_predict(C)
            groups = _merge_groupings(groups, g2)
    else:
        groups = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - merge_threshold, **kw).fit_predict(C)
    mapping = {tid: int(g) for tid, g in zip(ids, groups)}
    new = np.array([mapping.get(int(l), -1) if l >= 0 else -1 for l in labels], dtype=int)
    return _relabel(new)


def _merge_groupings(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Объединение двух разбиений: элементы, попавшие вместе хотя бы в одном, — вместе (union-find)."""
    parent = list(range(len(a)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for grouping in (a, b):
        first: dict[int, int] = {}
        for i, g in enumerate(grouping):
            if g in first:
                parent[find(i)] = find(first[g])
            else:
                first[g] = i
    roots = [find(i) for i in range(len(a))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots])


def centroids(E: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for tid in np.unique(labels[labels >= 0]):
        c = E[labels == tid].mean(axis=0)
        nrm = np.linalg.norm(c)
        out[int(tid)] = c / nrm if nrm else c
    return out


def _reassign_outliers(E: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    out_idx = np.where(labels == -1)[0]
    cents = centroids(E, labels)
    if not len(out_idx) or not cents:
        return labels
    ids = list(cents)
    C = np.stack([cents[i] for i in ids])
    sims = E[out_idx] @ C.T
    best = sims.argmax(axis=1)
    labels = labels.copy()
    for k, i in enumerate(out_idx):
        if sims[k, best[k]] >= threshold:
            labels[i] = ids[best[k]]
    return labels


def assign_new(E_new: np.ndarray, result: ClusterResult, threshold: float = 0.35) -> np.ndarray:
    """Новые тексты → ближайший кластер по центроиду (для инкрементальных выгрузок / запросов)."""
    ids = [t.id for t in result.topics if t.centroid is not None]
    if not ids:
        return np.full(len(E_new), -1)
    C = np.stack([result.topic(i).centroid for i in ids])
    sims = np.asarray(E_new) @ C.T
    best = sims.argmax(axis=1)
    return np.array([ids[b] if sims[k, b] >= threshold else -1 for k, b in enumerate(best)])


# ---------- описание кластеров ----------

def class_tfidf_keywords(texts: Sequence[str], labels: np.ndarray, n_keywords: int = 8, min_df: int = 2) -> dict[int, list[str]]:
    """c-TF-IDF (как в BERTopic): tf слова в классе × log(1 + средняя длина класса / частота слова по корпусу)."""
    from sklearn.feature_extraction.text import CountVectorizer
    ids = sorted(set(int(l) for l in labels if l >= 0))
    if not ids:
        return {}
    docs_by_class = [" ".join(texts[i] for i in np.where(labels == tid)[0]) for tid in ids]
    try:
        cv = CountVectorizer(analyzer=T.analyzer, min_df=1)
        X = cv.fit_transform(docs_by_class).astype(np.float64)
    except ValueError:
        return {tid: [] for tid in ids}
    vocab = np.array(cv.get_feature_names_out())
    total = np.asarray(X.sum(axis=0)).ravel()
    keep = total >= min_df
    tf = X.multiply(1.0 / np.maximum(X.sum(axis=1), 1))
    avg = X.sum() / len(ids)
    idf = np.log(1 + avg / np.maximum(total, 1))
    W = tf.multiply(idf).toarray()
    W[:, ~keep] = 0
    out = {}
    for row, tid in enumerate(ids):
        top = np.argsort(-W[row])[: n_keywords * 2]
        words = [vocab[j] for j in top if W[row, j] > 0]
        out[tid] = _dedupe_keywords(words, n_keywords)
    return out


def _dedupe_keywords(words: list[str], n: int) -> list[str]:
    """Биграмма вытесняет входящие в неё слова, если стоит выше; иначе слово вытесняет биграмму."""
    out: list[str] = []
    for w in words:
        parts = set(w.split())
        if any(parts <= set(o.split()) or set(o.split()) <= parts for o in out):
            continue
        out.append(w)
        if len(out) >= n:
            break
    return out


def split_large(E: np.ndarray, texts: Sequence[str], labels: np.ndarray, share: float = 0.15,
                random_state: int = 42) -> np.ndarray:
    """Вторая ступень: кластер больше `share` корпуса (например, «мешок» благодарностей на треть отзывов) ещё раз
    кластеризуется внутри себя UMAP → HDBSCAN (min_cluster_size = max(15, размер/40)); его тексты получают метки
    подкластеров, выбросы второй ступени остаются в общем подкластере. Остальные кластеры не меняются."""
    labels = np.asarray(labels).copy()
    n = len(labels)
    nxt = int(labels.max()) + 1 if n else 0
    for c in [c for c in np.unique(labels) if c >= 0 and (labels == c).sum() > share * n]:
        idx = np.where(labels == c)[0]
        sub = _umap_hdbscan(E[idx], max(15, len(idx) // 40), max(3, max(15, len(idx) // 40) // 3), 15, 5, random_state)
        if len(set(sub.tolist()) - {-1}) < 2:
            continue
        for s_ in sorted(set(sub.tolist())):
            labels[idx[sub == s_]] = nxt
            nxt += 1
    return _relabel(labels)


def describe_clusters(E: np.ndarray, texts: Sequence[str], labels: np.ndarray, n_keywords: int = 8,
                      n_representatives: int = 5) -> list[TopicInfo]:
    cents = centroids(E, labels)
    kws = class_tfidf_keywords(texts, labels, n_keywords=n_keywords)
    topics = []
    for tid in sorted(cents):
        idx = np.where(labels == tid)[0]
        sims = E[idx] @ cents[tid]
        order = idx[np.argsort(-sims)]
        reps, seen = [], set()
        for i in order:
            key = texts[i].strip().lower()
            if key and key not in seen:
                seen.add(key)
                reps.append(int(i))
            if len(reps) >= n_representatives:
                break
        topics.append(TopicInfo(tid, int(len(idx)), kws.get(tid, []), reps, centroid=cents[tid]))
    return topics


# ---------- оценка против разметки ----------

def evaluate(pred: np.ndarray, true: Sequence, ignore_outliers: bool = False) -> dict:
    """Кластеры против истинных меток.
    purity — доля документов, совпадающих с меткой большинства своего кластера (1 − purity = «доля нерелевантных»);
    bcubed_p / bcubed_r — B-cubed: для каждого документа доля «своих» (та же метка) в его кластере / доля «своих»,
    попавших в его кластер; усредняется по документам. P отвечает на «много ли чужого в кластере»,
    R — на «собираются ли похожие в одну группу». Выбросы (-1) в B-cubed считаются одиночными кластерами.
    clean_clusters — доля кластеров с чистотой ≥ 0.8."""
    from collections import Counter
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    pred = np.asarray(pred)
    true = np.asarray([str(t) for t in true])
    mask = pred >= 0 if ignore_outliers else np.ones(len(pred), dtype=bool)
    p, t = pred[mask], true[mask]
    out = {"n_clusters": int(len(np.unique(pred[pred >= 0]))), "outliers": float((pred == -1).mean()) if len(pred) else 0.0,
           "ari": 0.0, "nmi": 0.0, "purity": 0.0, "bcubed_p": 0.0, "bcubed_r": 0.0, "bcubed_f": 0.0, "clean_clusters": 0.0}
    if not len(p):
        return out
    out["ari"] = float(adjusted_rand_score(t, p))
    out["nmi"] = float(normalized_mutual_info_score(t, p))
    inl = pred >= 0
    hits, clean, n_cl = 0, 0, 0
    for cid in np.unique(pred[inl]):
        c = Counter(true[pred == cid])
        top = c.most_common(1)[0][1]
        hits += top
        n_cl += 1
        clean += top / sum(c.values()) >= 0.8
    out["purity"] = float(hits / max(inl.sum(), 1))
    out["clean_clusters"] = float(clean / max(n_cl, 1))
    # B-cubed; выбросы — синглтоны
    pp = pred.copy()
    nxt = pp.max() + 1 if len(pp) else 0
    for i in np.where(pp < 0)[0]:
        pp[i] = nxt
        nxt += 1
    pair = Counter(zip(pp, true))
    csize = Counter(pp)
    lsize = Counter(true)
    prec = np.array([pair[(pp[i], true[i])] / csize[pp[i]] for i in range(len(pp))])
    rec = np.array([pair[(pp[i], true[i])] / lsize[true[i]] for i in range(len(pp))])
    out["bcubed_p"], out["bcubed_r"] = float(prec.mean()), float(rec.mean())
    out["bcubed_f"] = float(2 * out["bcubed_p"] * out["bcubed_r"] / max(out["bcubed_p"] + out["bcubed_r"], 1e-9))
    return out


def knn_overlap(A: np.ndarray, B: np.ndarray, k: int = 10) -> float:
    """Доля общих k ближайших соседей документа в двух пространствах (например, эмбеддинги исходного и замусоренного
    текста). 1 — соседство не изменилось, ~k/n — случайное. Метрика не зависит от кластеризации."""
    na, nb = _knn(A, k), _knn(B, k)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(na, nb)]))


def knn_label_share(E: np.ndarray, labels: Sequence, target, k: int = 10) -> float:
    """Для документов с меткой target — средняя доля соседей с той же меткой (например, «SMM-формат»).
    Сравнивать с долей target в корпусе: превышение = эмбеддинг кодирует эту метку."""
    labels = np.asarray(labels)
    nn = _knn(E, k)
    idx = np.where(labels == target)[0]
    return float(np.mean([(labels[nn[i]] == target).mean() for i in idx])) if len(idx) else 0.0


def _knn(E: np.ndarray, k: int) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    n = len(E)
    out = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, 1024):
        S = E[s:s + 1024] @ E.T
        for j in range(S.shape[0]):
            S[j, s + j] = -np.inf
        out[s:s + 1024] = np.argpartition(-S, k, axis=1)[:, :k]
    return out
