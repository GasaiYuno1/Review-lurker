#!/usr/bin/env python3
"""Замер скорости моделей эмбеддингов на CPU в одиночку (без других процессов): 200 отзывов о банках (~960 символов)
и 400 тикетов (~80 символов); оценка времени на 40 000 тикетов и 3000 звонков.
→ research/results/speed_cpu.csv
python3 research/comments/speed.py [модель ...]"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RESULTS  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lurker.embed import ST_MODELS, get_embedder  # noqa: E402

torch.set_num_threads(os.cpu_count() or 4)
bank = pd.read_parquet(DATA / "comments/bank.parquet")["orig"].head(200).tolist()
tick = pd.read_parquet(DATA / "comments/tickets.parquet")["clean"].head(400).tolist()
rows = []
for name in (sys.argv[1:] or ["tfidf", "hybrid", *ST_MODELS]):
    t = time.time()
    m = get_embedder(name)
    if name in ("tfidf", "hybrid"):
        m.fit(tick + bank)
    load = time.time() - t
    m.encode(tick[:8])
    t = time.time(); m.encode(bank); tb = time.time() - t
    t = time.time(); m.encode(tick); tt = time.time() - t
    row = {"model": name, "load_s": round(load, 1), "reviews_per_s": round(200 / tb, 1), "tickets_per_s": round(400 / tt, 1),
           "min_40k_tickets": round(40000 / (400 / tt) / 60, 1), "cpu_threads": torch.get_num_threads()}
    rows.append(row)
    print(row, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "speed_cpu.csv", index=False)
