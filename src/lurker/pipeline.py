"""Оркестратор: нормализованная таблица → очистка → эмбеддинги → (предклассификация) → кластеры → названия → динамика → саммари.
Тяжёлые шаги кэшируются на диске по хэшу входа (очистка — sqlite по тексту, эмбеддинги и кластеры — по хэшу данных и параметров),
поэтому повторный запуск на том же файле мгновенный. LLM необязателен: без него названия тем — ключевые слова,
саммари — экстрактивные, период из вопроса — правилами.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .clean import LLMCleaner, RuleCleaner, TextCache, clean_texts
from .cluster import ClusterResult, cluster_embeddings
from .config import Config
from .dynamics import Finding, detect_changes, timeline
from .embed import get_embedder
from .preclass import Criteria, classify, classify_hybrid, label_prototypes
from .summarize import TopicSummary, name_topics, summarize_topics


@dataclass
class Analysis:
    df: pd.DataFrame                     # RangeIndex; колонки STANDARD + clean, topic, topic_name (+ label при предклассификации)
    E: np.ndarray                        # эмбеддинги, строка i ↔ df.iloc[i] (нули для пустых текстов)
    embedder: object
    clusters: ClusterResult
    summaries: list[TopicSummary]
    findings: list[Finding]
    names: dict[int, str]
    freq: str
    params: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    criteria: Optional[Criteria] = None

    def topic_name(self, tid: int) -> str:
        return self.names.get(tid, "прочее" if tid == -1 else str(tid))

    def timeline(self, freq: Optional[str] = None, filters: Optional[dict] = None, shares: bool = False,
                 topic_col: str = "topic_name") -> pd.DataFrame:
        return timeline(self.df, topic_col=topic_col, freq=freq or self.freq, filters=filters, shares=shares)


def _hash(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]


def run_analysis(df: pd.DataFrame, cfg: Optional[Config] = None, *, llm=None, freq: str = "W",
                 cache_dir: Optional[str | Path] = None, backend: Optional[str] = None, method: Optional[str] = None,
                 max_topics: Optional[int] = None, merge_threshold: Optional[float] = None,
                 min_cluster_size: Optional[int] = None, criteria: Optional[Criteria] = None,
                 llm_clean: bool = False, llm_names: bool = True, log: Optional[Callable[[str], None]] = None) -> Analysis:
    cfg = cfg or Config()
    log = log or (lambda s: None)
    backend = backend or cfg.embedding.backend
    method = method or cfg.cluster.method
    max_topics = cfg.cluster.max_topics if max_topics is None else max_topics
    merge_threshold = cfg.cluster.merge_threshold if merge_threshold is None else merge_threshold
    min_cluster_size = cfg.cluster.min_cluster_size if min_cluster_size is None else min_cluster_size
    cache_dir = Path(cache_dir) if cache_dir else (Path(cfg.cache.dir) if cfg.cache.dir else None)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    df = df.reset_index(drop=True).copy()
    df["text"] = df["text"].fillna("").astype(str)

    # 1. очистка
    t = time.time()
    cleaner = LLMCleaner(llm, batch_size=cfg.llm.batch_size) if (llm is not None and llm_clean) else RuleCleaner()
    df["clean"] = clean_texts(df["text"].tolist(), cleaner, TextCache(cache_dir / "clean.sqlite") if cache_dir else None)
    timings["clean"] = time.time() - t
    log(f"очистка ({cleaner.name}): {timings['clean']:.1f} с")

    # 2. эмбеддинги
    t = time.time()
    mask = (df["clean"].str.len() > 3).to_numpy()
    texts = df.loc[mask, "clean"].tolist()
    key = _hash(backend, cfg.embedding.model, *texts)
    E_sub = embedder = None
    if cache_dir and (cache_dir / f"emb-{key}.npy").is_file() and (cache_dir / f"emb-{key}.pkl").is_file():
        try:
            E_sub = np.load(cache_dir / f"emb-{key}.npy")
            embedder = pickle.loads((cache_dir / f"emb-{key}.pkl").read_bytes())
        except Exception:
            E_sub = embedder = None
    if E_sub is None:
        embedder = get_embedder(backend, model=cfg.embedding.model, device=cfg.embedding.device).fit(texts)
        E_sub = embedder.encode(texts) if texts else np.zeros((0, 1), dtype=np.float32)
        if cache_dir and texts:
            np.save(cache_dir / f"emb-{key}.npy", E_sub)
            try:
                (cache_dir / f"emb-{key}.pkl").write_bytes(pickle.dumps(embedder))
            except Exception:
                pass
    E = np.zeros((len(df), E_sub.shape[1] if E_sub.ndim == 2 else 1), dtype=np.float32)
    E[mask] = E_sub
    timings["embed"] = time.time() - t
    log(f"эмбеддинги ({getattr(embedder, 'name', backend)}, {E.shape[1]}d): {timings['embed']:.1f} с")

    # 3. предклассификация
    if criteria is not None and criteria.labels:
        t = time.time()
        protos = label_prototypes(criteria, embedder, texts, E_sub)
        if llm is not None:
            labels = classify_hybrid(texts, E_sub, criteria, protos, llm, threshold=cfg.preclass.threshold)
        else:
            labels, _ = classify(E_sub, criteria, protos, threshold=cfg.preclass.threshold, texts=texts)
        df["label"] = criteria.other
        df.loc[mask, "label"] = labels
        timings["preclass"] = time.time() - t
        log(f"предклассификация: {timings['preclass']:.1f} с, " + ", ".join(f"{k}: {v}" for k, v in df['label'].value_counts().items()))

    # 4. кластеры
    t = time.time()
    params = dict(method=method, min_cluster_size=min_cluster_size, max_topics=max_topics, merge_threshold=merge_threshold,
                  outlier_threshold=cfg.cluster.outlier_threshold)
    ckey = _hash(key, params)
    clusters = None
    if cache_dir and (cache_dir / f"clusters-{ckey}.pkl").is_file():
        try:
            clusters = pickle.loads((cache_dir / f"clusters-{ckey}.pkl").read_bytes())
        except Exception:
            clusters = None
    if clusters is None:
        clusters = cluster_embeddings(E_sub, texts, **params)
        if cache_dir:
            (cache_dir / f"clusters-{ckey}.pkl").write_bytes(pickle.dumps(clusters))
    # индексы представителей — в координатах df
    pos = np.where(mask)[0]
    for tp in clusters.topics:
        tp.representatives = [int(pos[i]) for i in tp.representatives]
    df["topic"] = -1
    df.loc[mask, "topic"] = clusters.labels
    timings["cluster"] = time.time() - t
    log(f"кластеры ({clusters.method}): {len(clusters.topics)} тем, прочее {clusters.n_outliers}, {timings['cluster']:.1f} с")

    # 5. названия
    t = time.time()
    names: dict[int, str] = {}
    if llm is not None and llm_names and clusters.topics:
        try:
            names = name_topics(clusters, df["clean"].tolist(), llm)
        except Exception:
            names = {}
    for tp in clusters.topics:
        names.setdefault(tp.id, tp.label)
    names[-1] = "прочее"
    df["topic_name"] = df["topic"].map(names)
    timings["names"] = time.time() - t

    # 6. динамика и саммари
    t = time.time()
    findings = detect_changes(timeline(df, topic_col="topic", freq=freq))
    summaries = summarize_topics(df, clusters, findings, topic_col="topic")
    for s in summaries:
        s.name = names.get(s.id, s.name)
    timings["dynamics"] = time.time() - t
    log(f"динамика: {len(findings)} находок")

    return Analysis(df, E, embedder, clusters, summaries, findings, names, freq,
                    params=dict(params, backend=backend, cleaner=cleaner.name), timings=timings, criteria=criteria)


def save(an: Analysis, path: str | Path) -> None:
    Path(path).write_bytes(pickle.dumps(an))


def load(path: str | Path) -> Analysis:
    return pickle.loads(Path(path).read_bytes())
