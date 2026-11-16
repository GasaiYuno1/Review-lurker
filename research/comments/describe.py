#!/usr/bin/env python3
"""Задача 1, интерпретация кластеров (ТЗ v2): для значимых кластеров — ключевые слова (c-TF-IDF), представительные
тексты и описание через LLM по 10–30 текстам (название, описание, тональность, однородность, выбивающиеся тексты).
Плюс лист ручной проверки: по 10 случайных текстов каждого кластера — для оценки доли нерелевантных глазами.
LLM — любой OpenAI-совместимый endpoint из конфига (по умолчанию локальный llama.cpp с Qwen3-4B, configs/llamacpp.yaml).
python3 research/comments/describe.py --dataset bank --variant clean --model user-base [--config configs/llamacpp.yaml]
        [--min-share 0.01] [--max-clusters 25] [--n-texts 20] [--no-llm]
→ research/results/clusters_<dataset>.json, research/results/manual_review_<dataset>.md
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
from common import BUILD, DATA, RESULTS, ROOT  # noqa: E402
from embed import emb_path  # noqa: E402
from lurker.config import load_config  # noqa: E402
from lurker.llm import LLMClient  # noqa: E402
from lurker.summarize import describe_cluster  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bank")
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--model", default="user-base")
    ap.add_argument("--config", default=str(ROOT / "configs/llamacpp.yaml"))
    ap.add_argument("--min-share", type=float, default=0.01)
    ap.add_argument("--max-clusters", type=int, default=25)
    ap.add_argument("--n-texts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--mcs", type=int, default=0, help="перекластеризовать с этим min_cluster_size (0 — взять кластеры эксперимента)")
    a = ap.parse_args()
    df = pd.read_parquet(DATA / "comments" / f"{a.dataset}.parquet")
    key = f"{a.dataset}__{a.variant}__{a.model}"
    E = np.load(emb_path(a.dataset, a.variant, a.model))
    texts = df[a.variant].fillna("").astype(str).tolist()
    if a.mcs:
        from lurker.cluster import cluster_embeddings
        res = cluster_embeddings(E, texts, method="umap_hdbscan", min_cluster_size=a.mcs)
        key += f"__mcs{a.mcs}"
        lab = res.labels
        info = {t.id: {"id": t.id, "size": t.size, "keywords": t.keywords, "reps": t.representatives} for t in res.topics}
        np.save(BUILD / "clusters" / f"{key}.npy", lab)
    else:
        lab = np.load(BUILD / "clusters" / f"{key}.npy")
        info = {c["id"]: c for c in json.loads((BUILD / "clusters" / f"{key}.json").read_text(encoding="utf-8"))}
    llm = None if a.no_llm else LLMClient(load_config(a.config).llm, cache_dir=str(ROOT / ".cache"))
    rng = np.random.default_rng(a.seed)
    ids = [c for c in sorted(info, key=lambda c: -info[c]["size"]) if info[c]["size"] >= a.min_share * len(df)][: a.max_clusters]
    out, sheet = [], [f"# Ручная проверка кластеров: {a.dataset} / {a.variant} / {a.model}", "",
                      "По 10 случайных текстов каждого кластера. Отметка: + подходит под описание, − не подходит.", ""]
    for cid in ids:
        idx = np.where(lab == cid)[0]
        c = E[idx].mean(axis=0)
        near = idx[np.argsort(-(E[idx] @ c))][: a.n_texts // 2]                       # ближе к центру
        rest = np.setdiff1d(idx, near)
        rnd = rng.choice(rest, min(len(rest), a.n_texts - len(near)), replace=False) if len(rest) else []
        sample = list(near) + list(rnd)                                                # + случайные: честнее для «однородности»
        rec = {"id": int(cid), "size": int(len(idx)), "share": round(len(idx) / len(df), 4), "keywords": info[cid]["keywords"][:8],
               "sample_docs": [int(i) for i in sample]}                                  # какие тексты видела LLM (для проверки «выбивающихся»)
        if "rating" in df and (df["rating"] > 0).any():
            r = df["rating"].to_numpy()[idx]
            rec["rating_mean"] = round(float(r.mean()), 2)
            rec["neg_share"] = round(float((r <= 2).mean()), 3)
        if df["label"].astype(bool).any():
            vc = df["label"].to_numpy()[idx]
            top = pd.Series(vc).value_counts()
            rec["top_label"], rec["top_label_share"] = str(top.index[0]), round(float(top.iloc[0] / len(idx)), 3)
        if llm is not None:
            t = time.time()
            try:
                rec.update(describe_cluster([texts[i] for i in sample], llm, rec["keywords"]))
                rec["llm_outlier_share"] = round(len(rec.get("outliers", [])) / len(sample), 3)
            except Exception as e:
                rec["llm_error"] = str(e)[:200]
            rec["llm_s"] = round(time.time() - t, 1)
        out.append(rec)
        print(f"{cid:3d} {len(idx):5d}  {rec.get('name', '—'):40.40}  ← {', '.join(rec['keywords'][:5])}", flush=True)
        manual = rng.choice(idx, min(10, len(idx)), replace=False)
        sheet += [f"## Кластер {cid}: {rec.get('name') or ', '.join(rec['keywords'][:4])} ({len(idx)} текстов)", "",
                  f"Описание: {rec.get('description', '—')}", ""]
        sheet += [f"- [ ] {' '.join(texts[i].split())[:300]}" for i in manual] + [""]
        (RESULTS / f"clusters_{a.dataset}.json").write_text(json.dumps({"key": key, "clusters": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS / f"manual_review_{a.dataset}.md").write_text("\n".join(sheet), encoding="utf-8")
    # для ручной проверки: те же 10 случайных текстов на кластер в машиночитаемом виде
    rows = []
    rng2 = np.random.default_rng(a.seed + 1)
    for rec in out:
        idx = np.where(lab == rec["id"])[0]
        for i in rng2.choice(idx, min(10, len(idx)), replace=False):
            rows.append({"cluster": rec["id"], "name": rec.get("name", ""), "description": rec.get("description", ""), "doc": int(i),
                         "text": " ".join(texts[i].split())[:600], "label": df["label"].iloc[i] if "label" in df else ""})
    pd.DataFrame(rows).to_csv(RESULTS / f"manual_sample_{a.dataset}.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
