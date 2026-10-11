#!/usr/bin/env python3
"""Задача 1: эмбеддинги для всех (набор × вариант текста × модель), с кэшем в build/research/emb/.
Порядок — от важного к дорогому: сначала лёгкие модели на всех наборах, потом тяжёлые на коротких наборах.
Время каждой пары пишется в research/results/embed_timings.csv (для оценки масштаба 40 тыс. тикетов).
python3 research/comments/embed.py [--only dataset[,dataset]] [--models m1,m2]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BUILD, DATA, RESULTS  # noqa: E402
from lurker.embed import get_embedder  # noqa: E402

LIGHT = ["tfidf", "hybrid", "rubert-tiny2", "e5-small", "user-base"]
HEAVY = ["e5-base", "user-bge-m3", "frida"]
VARIANTS = {"bank": ["orig", "dirty", "clean"], "geo": ["orig", "dirty", "clean"], "tickets": ["dirty", "clean"]}
# FRIDA (823M) на CPU ~1 отзыв/с: только тикеты (короткие), иначе полтора часа на один набор
PLAN = ([(d, m) for m in LIGHT for d in ("tickets", "geo", "bank")] +
        [(d, m) for m in ("e5-base", "user-bge-m3") for d in ("tickets", "geo")] + [("tickets", "frida")])


def emb_path(dataset: str, variant: str, model: str) -> Path:
    return BUILD / "emb" / dataset / f"{variant}__{model}.npy"


def load_texts(dataset: str, variant: str) -> list[str]:
    df = pd.read_parquet(DATA / "comments" / f"{dataset}.parquet")
    return [t if t and t.strip() else "-" for t in df[variant].fillna("").astype(str)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    models = set(a.models.split(",")) if a.models else None
    log = RESULTS / "embed_timings.csv"
    new_log = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["dataset", "variant", "model", "n", "mean_chars", "seconds", "texts_per_sec"])
        for dataset, model in PLAN:
            if (only and dataset not in only) or (models and model not in models):
                continue
            todo = [v for v in VARIANTS[dataset] if not emb_path(dataset, v, model).exists()]
            if not todo:
                continue
            t0 = time.time()
            emb = get_embedder(model) if model not in ("tfidf", "hybrid") else None
            load_s = time.time() - t0
            for v in todo:
                texts = load_texts(dataset, v)
                t = time.time()
                e = get_embedder(model).fit(texts) if emb is None else emb
                E = e.encode(texts).astype(np.float32)
                dt = time.time() - t
                p = emb_path(dataset, v, model)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, E)
                w.writerow([dataset, v, model, len(texts), round(float(np.mean([len(x) for x in texts])), 1), round(dt, 1), round(len(texts) / dt, 2)])
                f.flush()
                print(f"{dataset:8} {v:6} {model:12} {E.shape} {dt:7.1f} с ({len(texts) / dt:.1f} текст/с; загрузка модели {load_s:.0f} с)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
