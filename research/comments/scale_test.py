#!/usr/bin/env python3
"""Масштаб: 40 000 синтетических тикетов (реальная выборка заказчика — 40 тыс.) целиком через пайплайн на CPU:
генерация → очистка → эмбеддинги (hybrid и USER-base) → UMAP + HDBSCAN → метрики против скрытых тем.
→ research/results/scale_40k.csv"""
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from common import RESULTS  # noqa: E402
import pandas as pd  # noqa: E402
import gen_testdata as g  # noqa: E402
from lurker.clean import RuleCleaner  # noqa: E402
from lurker.cluster import cluster_embeddings, evaluate  # noqa: E402
from lurker.embed import get_embedder  # noqa: E402

rows_out = []
t = time.time()
rows, labels = g.generate(40000, 11, date(2025, 9, 1), date(2026, 9, 7))
texts = [r["Комментарий"] for r in rows]
y = [l["topic"] for l in labels]
gen_s = time.time() - t
t = time.time()
rc = RuleCleaner()
clean = [rc.clean(x).text for x in texts]
clean_s = time.time() - t
keep = [i for i, c in enumerate(clean) if len(c) > 3]
clean = [clean[i] for i in keep]
y = [y[i] for i in keep]
print(f"тикетов {len(texts)}, после очистки непустых {len(clean)}; генерация {gen_s:.0f} с, очистка {clean_s:.0f} с", flush=True)
for model in ("hybrid", "user-base"):
    t = time.time()
    emb = get_embedder(model).fit(clean)
    E = emb.encode(clean)
    emb_s = time.time() - t
    t = time.time()
    res = cluster_embeddings(E, clean, method="umap_hdbscan")
    cl_s = time.time() - t
    ev = evaluate(res.labels, y)
    row = {"n": len(clean), "model": model, "clean_s": round(clean_s), "embed_s": round(emb_s), "cluster_s": round(cl_s),
           **{k: round(v, 4) for k, v in ev.items()}}
    rows_out.append(row)
    print(row, flush=True)
    pd.DataFrame(rows_out).to_csv(RESULTS / "scale_40k.csv", index=False)
