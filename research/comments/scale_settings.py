#!/usr/bin/env python3
"""Задача 1, масштаб 40 тыс. (реальный объём заказчика): рекомендованный конвейер целиком и настройка под объём.
На 40 тыс. с min_cluster_size = √n/2 (≈ 100) темы дробятся на подтемы (119 кластеров на 25 тем, scale_40k.csv).
Здесь: 40 000 синтетических тикетов (тот же генератор и зерно, что в scale_test.py) → удаление шаблонов → очистка →
USER-bge-m3 (рекомендованная модель для коротких тикетов; время замеряется, раньше было только оценено) → сетка
настроек: min_cluster_size и слияние близких кластеров по центроидам (lurker.cluster.reduce_topics).
python3 research/comments/scale_settings.py → research/results/scale_40k_settings.csv (эмбеддинги кэшируются в build/)
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from common import BUILD, RESULTS  # noqa: E402
import gen_testdata as g  # noqa: E402
from lurker.clean import RuleCleaner, remove_boilerplate  # noqa: E402
from lurker.cluster import cluster_embeddings, evaluate, reduce_topics  # noqa: E402
from lurker.embed import get_embedder  # noqa: E402

MODEL = "user-bge-m3"


def main() -> int:
    out = RESULTS / "scale_40k_settings.csv"
    rows = []
    t = time.time()
    data, labels = g.generate(40000, 11, date(2025, 9, 1), date(2026, 9, 7))
    texts = [r["Комментарий"] for r in data]
    y = [l["topic"] for l in labels]
    t = time.time()
    pre, removed = remove_boilerplate(texts)
    rc = RuleCleaner()
    clean = [rc.clean(x).text for x in pre]
    clean_s = time.time() - t
    keep = [i for i, c in enumerate(clean) if len(c) > 3]
    clean, y = [clean[i] for i in keep], [y[i] for i in keep]
    print(f"тикетов {len(texts)}, непустых после очистки {len(clean)}; шаблонов {len(removed)}; очистка {clean_s:.0f} с", flush=True)
    p = BUILD / "emb" / "scale40k" / f"clean_bp__{MODEL}.npy"
    t = time.time()
    if p.exists():
        E = np.load(p)
        emb_s = float("nan")
    else:
        E = get_embedder(MODEL).encode(clean).astype(np.float32)
        emb_s = time.time() - t
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, E)
    print(f"эмбеддинги {MODEL}: {emb_s:.0f} с", flush=True)
    n = len(clean)
    for mcs in (100, 200, 400):
        t = time.time()
        res = cluster_embeddings(E, clean, method="umap_hdbscan", min_cluster_size=mcs)
        cl_s = time.time() - t
        for merge in (None, 0.9, 0.85, 0.8):
            lab = reduce_topics(E, res.labels, merge_threshold=merge) if merge else res.labels
            ev = evaluate(lab, y)
            rows.append({"n": n, "model": MODEL, "min_cluster_size": mcs, "merge_threshold": merge or "", "clean_s": round(clean_s),
                         "embed_s": round(emb_s) if emb_s == emb_s else "", "cluster_s": round(cl_s),
                         **{k: round(v, 4) for k, v in ev.items()}})
            print({k: rows[-1][k] for k in ("min_cluster_size", "merge_threshold", "n_clusters", "nmi", "bcubed_p", "bcubed_r",
                                            "purity", "clean_clusters")}, flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
