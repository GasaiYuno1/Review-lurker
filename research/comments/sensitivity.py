#!/usr/bin/env python3
"""Задача 1: чувствительность кластеризации к настройкам (ТЗ: BERTopic «из коробки» дал мусор — проверить с
нормальной настройкой). На наборах с метками (geo, tickets), одна модель, очищенный текст:
  UMAP n_neighbors × HDBSCAN min_cluster_size; без UMAP (HDBSCAN прямо по эмбеддингам); KMeans и агломеративная;
  «как из коробки» — параметры BERTopic по умолчанию (UMAP 15/5, HDBSCAN min_cluster_size=10) без дораспределения выбросов.
→ research/results/sensitivity.csv
python3 research/comments/sensitivity.py [--model user-base] [--datasets geo,tickets]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS  # noqa: E402
from embed import emb_path  # noqa: E402
from lurker.cluster import cluster_embeddings, evaluate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="user-base")
    ap.add_argument("--datasets", default="geo,tickets")
    ap.add_argument("--variant", default="clean")
    a = ap.parse_args()
    rows = []
    for ds in a.datasets.split(","):
        df = pd.read_parquet(DATA / "comments" / f"{ds}.parquet")
        E = np.load(emb_path(ds, a.variant, a.model))
        texts = df[a.variant].fillna("").astype(str).tolist()
        y = df["label"].tolist()
        n = len(E)
        grid = [("bertopic-default", dict(method="umap_hdbscan", n_neighbors=15, n_components=5, min_cluster_size=10,
                                          min_samples=10, outlier_threshold=2.0))]
        for nn in (10, 15, 30, 50):
            for mcs in (10, 20, 32, 60, 100):
                grid.append((f"umap{nn}-mcs{mcs}", dict(method="umap_hdbscan", n_neighbors=nn, min_cluster_size=mcs)))
        grid.append(("umap15-mcs32-no-reassign", dict(method="umap_hdbscan", n_neighbors=15, min_cluster_size=32, outlier_threshold=2.0)))
        grid.append(("agglomerative-0.55", dict(method="agglomerative", distance_threshold=0.55)))
        for name, kw in grid:
            t = time.time()
            res = cluster_embeddings(E, texts, **kw)
            ev = evaluate(res.labels, y)
            rows.append({"dataset": ds, "model": a.model, "config": name, "seconds": round(time.time() - t, 1),
                         **{k: round(v, 4) for k, v in ev.items()}})
            print(f"{ds:8} {name:28} кластеров {ev['n_clusters']:3d} выбросы {ev['outliers']:.2f} NMI {ev['nmi']:.3f} "
                  f"B3-P {ev['bcubed_p']:.3f} B3-R {ev['bcubed_r']:.3f} чистота {ev['purity']:.3f}", flush=True)
        # «без UMAP»: HDBSCAN прямо по эмбеддингам (косинус через L2 на нормированных векторах)
        import hdbscan
        for mcs in (20, 32):
            t = time.time()
            lab = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=max(3, mcs // 3), metric="euclidean").fit_predict(E)
            ev = evaluate(lab, y)
            rows.append({"dataset": ds, "model": a.model, "config": f"hdbscan-no-umap-mcs{mcs}", "seconds": round(time.time() - t, 1),
                         **{k: round(v, 4) for k, v in ev.items()}})
            print(f"{ds:8} hdbscan-no-umap-mcs{mcs:<10} кластеров {ev['n_clusters']:3d} выбросы {ev['outliers']:.2f} NMI {ev['nmi']:.3f}", flush=True)
        from sklearn.cluster import KMeans
        k0 = df["label"].nunique()
        for k in (k0, 2 * k0):
            lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(E)
            ev = evaluate(lab, y)
            rows.append({"dataset": ds, "model": a.model, "config": f"kmeans-{k}", "seconds": 0,
                         **{kk: round(v, 4) for kk, v in ev.items()}})
            print(f"{ds:8} kmeans-{k:<22} NMI {ev['nmi']:.3f} B3-P {ev['bcubed_p']:.3f} B3-R {ev['bcubed_r']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "sensitivity.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
