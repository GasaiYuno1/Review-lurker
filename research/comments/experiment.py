#!/usr/bin/env python3
"""Задача 1: кластеризация и метрики для всех посчитанных эмбеддингов (набор × вариант текста × модель).

Что меряем (research/results/comments_metrics.csv, одна строка на тройку):
  устойчивость к мусору (bank, geo — есть исходный текст):
    knn_overlap   — доля общих 10 ближайших соседей с исходным текстом (1 — мусор не изменил соседство);
    ari_vs_orig   — согласие кластеров варианта с кластерами исходного текста;
    smm_nn_share  — у документов в SMM-формате доля SMM-соседей (в корпусе SMM 40%; выше — эмбеддинг «видит» формат);
    format_docs   — доля документов в «форматных» кластерах (≥ 80% SMM), кластеризация по мусору, а не по смыслу;
  качество групп (geo — метка рубрика, tickets — скрытая тема):
    purity, bcubed_p («сколько чужих в моём кластере»), bcubed_r («сколько своих со мной»), nmi, ari,
    clean_clusters (доля кластеров с чистотой ≥ 0.8), outliers, n_clusters; то же для KMeans(k = число меток);
  тональность (bank, geo): rating_nmi — насколько кластеры разделяют негатив/нейтраль/позитив.
Кластеризация: UMAP (15 соседей, 5 измерений, косинус) → HDBSCAN (min_cluster_size = max(8, √n/2)) → выбросы
к ближайшему центроиду при сходстве ≥ 0.35 (lurker.cluster.cluster_embeddings, схема BERTopic).
Кластеры и ключевые слова сохраняются в build/research/clusters/ для ноутбука и описаний.
python3 research/comments/experiment.py [--only geo] [--models ...] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, DATA, RESULTS  # noqa: E402
from embed import VARIANTS, emb_path  # noqa: E402
from lurker.cluster import cluster_embeddings, evaluate, knn_label_share, knn_overlap  # noqa: E402

MODELS = ["tfidf", "hybrid", "rubert-tiny2", "e5-small", "user-base", "e5-base", "user-bge-m3", "frida"]
OUT = RESULTS / "comments_metrics.csv"
CL = BUILD / "clusters"


def rating_class(r: pd.Series) -> np.ndarray:
    return np.where(r <= 2, "neg", np.where(r >= 4, "pos", "neu"))


def run_one(dataset: str, df: pd.DataFrame, variant: str, model: str, orig_cache: dict) -> dict:
    E = np.load(emb_path(dataset, variant, model))
    texts = df[variant].fillna("").astype(str).tolist()
    row = {"dataset": dataset, "model": model, "variant": variant, "n": len(E), "dim": E.shape[1]}
    t = time.time()
    res = cluster_embeddings(E, texts, method="umap_hdbscan")
    row["cluster_s"] = round(time.time() - t, 1)
    lab = res.labels
    CL.mkdir(parents=True, exist_ok=True)
    np.save(CL / f"{dataset}__{variant}__{model}.npy", lab)
    (CL / f"{dataset}__{variant}__{model}.json").write_text(json.dumps(
        [{"id": tp.id, "size": tp.size, "keywords": tp.keywords, "reps": tp.representatives} for tp in res.topics],
        ensure_ascii=False), encoding="utf-8")
    row["n_clusters"] = len(res.topics)
    row["outliers"] = round(float((lab == -1).mean()), 4)
    has_orig = "orig" in VARIANTS[dataset]
    if has_orig and variant != "orig" and (dataset, model) in orig_cache:
        Eo, lo = orig_cache[(dataset, model)]
        row["knn_overlap"] = round(knn_overlap(E, Eo), 4)
        both = (lab >= 0) & (lo >= 0)
        from sklearn.metrics import adjusted_rand_score
        row["ari_vs_orig"] = round(float(adjusted_rand_score(lo[both], lab[both])), 4)
    kind = df["dirty_kind"].astype(str).to_numpy()
    target = "smm" if dataset != "tickets" else "SMM"
    row["smm_share"] = round(float((kind == target).mean()), 4)
    row["smm_nn_share"] = round(knn_label_share(E, kind, target), 4)
    fmt_docs = 0
    for cid in np.unique(lab[lab >= 0]):
        m = lab == cid
        if (kind[m] == target).mean() >= 0.8:
            fmt_docs += m.sum()
    row["format_docs"] = round(fmt_docs / len(lab), 4)
    if df["label"].astype(bool).any():
        ev = evaluate(lab, df["label"].tolist())
        row.update({k: round(v, 4) for k, v in ev.items() if k not in ("n_clusters", "outliers")})
        from sklearn.cluster import KMeans
        k = df["label"].nunique()
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(E)
        evk = evaluate(km, df["label"].tolist())
        row.update({f"km_{m_}": round(evk[m_], 4) for m_ in ("nmi", "purity", "bcubed_p", "bcubed_r")})
    if (df["rating"].astype(float) > 0).any():
        from sklearn.metrics import normalized_mutual_info_score
        rc = rating_class(df["rating"].astype(float))
        row["rating_nmi"] = round(float(normalized_mutual_info_score(rc[lab >= 0], lab[lab >= 0])), 4)
    if variant == "orig":
        orig_cache[(dataset, model)] = (E, lab)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    done = pd.read_csv(OUT) if OUT.exists() and not a.force else pd.DataFrame()
    have = set(zip(done.get("dataset", []), done.get("model", []), done.get("variant", [])))
    rows = done.to_dict("records") if len(done) else []
    datasets = a.only.split(",") if a.only else list(VARIANTS)
    models = a.models.split(",") if a.models else MODELS
    for dataset in datasets:
        df = pd.read_parquet(DATA / "comments" / f"{dataset}.parquet")
        orig_cache: dict = {}
        for model in models:
            for variant in VARIANTS[dataset]:   # orig идёт первым — нужен как эталон
                if not emb_path(dataset, variant, model).exists():
                    continue
                if (dataset, model, variant) in have and variant != "orig":
                    continue
                if (dataset, model, variant) in have and variant == "orig":
                    E = np.load(emb_path(dataset, variant, model))
                    orig_cache[(dataset, model)] = (E, np.load(CL / f"{dataset}__orig__{model}.npy"))
                    continue
                row = run_one(dataset, df, variant, model, orig_cache)
                rows = [r for r in rows if (r["dataset"], r["model"], r["variant"]) != (dataset, model, variant)] + [row]
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print({k: row[k] for k in row if k in ("dataset", "model", "variant", "n_clusters", "outliers", "knn_overlap",
                                                       "ari_vs_orig", "smm_nn_share", "format_docs", "purity", "bcubed_p",
                                                       "bcubed_r", "nmi", "rating_nmi", "cluster_s")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
