#!/usr/bin/env python3
"""Задача 2, дополнительные проверки на кэшированных эмбеддингах (после run_anomaly.py):
  1) темы нужны? оценка по репликам внутри темы против той же оценки по всему корпусу (без тем);
  2) центрирование эмбеддингов (вычесть общий «жанровый» компонент) — помогает ли;
  3) параметры оценки по репликам: k соседей, по одной или двум худшим репликам;
  4) почему эмбеддинг целого звонка не видит подмену: сходство подменённого звонка с его исходной версией против
     сходства двух разных звонков одной темы.
→ research/results/calls_extra.csv, research/results/calls_splice_sanity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, DATA, RESULTS  # noqa: E402
from lurker.anomaly import assign_topics, knn_score, ranking_metrics, robust_z, turn_score  # noqa: E402
from lurker.cluster import cluster_embeddings  # noqa: E402
from run_anomaly import EMB, stt, texts_for  # noqa: E402

TYPES = ("splice", "splice_near", "rude", "absurd")


def evaluate(z, kind, types):
    anom, corr, nat = kind == "anomaly", kind == "corrupted", kind == "natural"
    pool = ~nat
    m = ranking_metrics(z[pool], anom[pool], (120,), corr[pool])
    row = {"ap": m["ap"], "auc": m["auc"], "p@120": m["p@120"], "noise@120": m["noise@120"]}
    for ty in TYPES:
        keep = pool & ((types == ty) | ~anom)
        row[f"ap_{ty}"] = ranking_metrics(z[keep], (types == ty)[keep], (30,))["ap"]
    row["ap_natural"] = ranking_metrics(z[kind != "anomaly"], nat[kind != "anomaly"], (50,))["ap"]
    return row


def center(E):
    E = E - E.mean(0)
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)


def main() -> int:
    df = pd.read_parquet(DATA / "calls" / "calls.parquet")
    kind, types = df["kind"].to_numpy(), df["anomaly_type"].to_numpy()
    docs, turns, owner, speaker = texts_for(df, "stt")
    op = speaker == "op"
    rows = []
    models = [p.name.split("__")[-1][:-4] for p in sorted(EMB.glob("stt__dialog__*.npy"))]
    for model in models:
        E0, T0 = np.load(EMB / f"stt__dialog__{model}.npy"), np.load(EMB / f"stt__allturns__{model}.npy")
        top = pd.read_parquet(BUILD / f"calls_scores__stt__{model}.parquet")["topic"].to_numpy()
        S = E0 @ E0.T
        np.fill_diagonal(S, np.nan)
        base = {"model": model, "mean_pair_cos": round(float(np.nanmean(S)), 3)}
        for name, s, t in (("turns в теме (k=10, 1 реплика)", turn_score(T0[op], owner[op], top), top),
                           ("turns без тем", turn_score(T0[op], owner[op], np.zeros(len(df), int)), np.zeros(len(df), int)),
                           ("turns в теме, k=3", turn_score(T0[op], owner[op], top, k=3), top),
                           ("turns в теме, k=30", turn_score(T0[op], owner[op], top, k=30), top),
                           ("turns в теме, 2 реплики", turn_score(T0[op], owner[op], top, top=2), top),
                           ("turns в теме, все реплики", turn_score(T0, owner, top), top),
                           ("knn звонка в теме", knn_score(E0, top), top),
                           ("knn звонка без тем", knn_score(E0, np.zeros(len(df), int)), np.zeros(len(df), int))):
            rows.append({**base, "check": name, **{k: round(v, 4) for k, v in evaluate(robust_z(s, t), kind, types).items()}})
        Ec, Tc = center(E0), center(T0)
        res = cluster_embeddings(Ec, docs, method="umap_hdbscan", outlier_threshold=-1.0)
        topc = assign_topics(Ec, res.labels)
        for name, s in (("центрирование: turns в теме", turn_score(Tc[op], owner[op], topc)), ("центрирование: knn звонка", knn_score(Ec, topc))):
            rows.append({**base, "check": name, **{k: round(v, 4) for k, v in evaluate(robust_z(s, topc), kind, types).items()}})
        pd.DataFrame(rows).to_csv(RESULTS / "calls_extra.csv", index=False)
        print(f"{model}: готово", flush=True)
    # 4) подмена против исходника (rubert-tiny2 — быстро)
    from prepare_calls import load_dialogs
    from lurker.embed import get_embedder
    full = load_dialogs()
    base_df = full[full["politeness"] >= 6].reset_index(drop=True)
    sp = df.index[df["anomaly_type"] == "splice"].tolist()
    san = []
    for model in [m for m in ("rubert-tiny2", "e5-small", "user-base") if m in models]:
        E = np.load(EMB / f"stt__dialog__{model}.npy")
        top = pd.read_parquet(BUILD / f"calls_scores__stt__{model}.parquet")["topic"].to_numpy()
        orig = ["\n".join(("оператор: " if s_ == "op" else "клиент: ") + stt(x) for s_, x in base_df.loc[i, "turns"]) for i in sp]
        Eo = get_embedder(model).encode(orig)
        cos_self = (E[sp] * Eo).sum(1)
        normal = np.where(kind == "normal")[0]
        rng = np.random.default_rng(0)
        same = [E[i] @ E[rng.choice(normal[(top[normal] == top[i]) & (normal != i)])] for i in rng.choice(normal, 400)
                if ((top[normal] == top[i]) & (normal != i)).any()]
        san.append({"model": model, "cos_spliced_vs_own_original_median": round(float(np.median(cos_self)), 3),
                    "cos_two_normal_same_topic_median": round(float(np.median(same)), 3),
                    "cos_two_normal_same_topic_p10": round(float(np.percentile(same, 10)), 3)})
    pd.DataFrame(san).to_csv(RESULTS / "calls_splice_sanity.csv", index=False)
    print(pd.DataFrame(san).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
