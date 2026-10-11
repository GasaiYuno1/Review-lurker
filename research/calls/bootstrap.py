#!/usr/bin/env python3
"""Задача 2, проверки выводов (аудит):
  1) доверительные интервалы: бутстреп по звонкам (1000 повторов) для главных метрик выбранной конфигурации и для
     базовой схемы ТЗ (расстояние до центра темы); пул — как в run_anomaly.py (подмешанные аномалии против нормальных
     и испорченных, естественно невежливые исключены) → research/results/calls_bootstrap.csv;
  2) находит ли оценка «естественно слабые» звонки: среди НЕТРОНУТЫХ звонков — связь z с оценками работы оператора
     из разметки самого датасета (0–10: вежливость, регламент, эффективность, результат, эмпатия, грамотность)
     → research/results/calls_grades.csv.
python3 research/calls/bootstrap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BUILD, DATA, RESULTS  # noqa: E402


def metrics(z, pos, cor):
    o = np.argsort(-z)
    return {"ap": average_precision_score(pos, z), "auc": roc_auc_score(pos, z), "p@40": pos[o[:40]].mean(),
            "p@120": pos[o[:120]].mean(), "noise@120": cor[o[:120]].mean(),
            "flagged": (z >= 2).mean(), "precision_z2": pos[z >= 2].mean() if (z >= 2).any() else np.nan,
            "recall_z2": (z[pos] >= 2).mean()}


def main() -> int:
    meta = pd.read_parquet(DATA / "calls" / "calls.parquet")[["id", "kind"]]
    rows = []
    rng = np.random.default_rng(0)
    for model in ("user-base", "user-bge-m3", "tfidf"):
        p = BUILD / f"calls_scores__stt__{model}.parquet"
        if not p.exists():
            continue
        s = pd.read_parquet(p).merge(meta, on="id")
        s = s[s.kind != "natural"]
        pos, cor = (s.kind == "anomaly").to_numpy(), (s.kind == "corrupted").to_numpy()
        for method in ("turns_c", "centroid"):
            z = s[method].to_numpy()
            point = metrics(z, pos, cor)
            boot = [metrics(z[i], pos[i], cor[i]) for i in (rng.integers(0, len(z), len(z)) for _ in range(1000))]
            for k, v in point.items():
                vals = np.array([b[k] for b in boot], dtype=float)
                rows.append({"model": model, "method": method, "metric": k, "value": round(float(v), 4),
                             "ci_low": round(float(np.nanpercentile(vals, 2.5)), 4), "ci_high": round(float(np.nanpercentile(vals, 97.5)), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "calls_bootstrap.csv", index=False)
    print(out[out.metric.isin(["ap", "auc", "p@40", "p@120", "precision_z2", "recall_z2", "flagged"])].to_string(index=False))
    grades()
    return 0


def grades() -> None:
    calls = pd.read_parquet(DATA / "calls" / "calls.parquet")
    s = pd.read_parquet(BUILD / "calls_scores__stt__user-base.parquet")[["id", "turns_c", "centroid"]].merge(calls, on="id")
    normal = s[s.kind == "normal"]
    rows = []
    for col in ("politeness", "regulation", "efficiency", "result", "empathy", "literacy"):
        x = normal[col].astype(float)
        ok = x.notna()
        low = (x[ok] <= 6).to_numpy()
        top = normal[ok].sort_values("turns_c", ascending=False)
        rows.append({"grade": col, "low_share": round(float(low.mean()), 3),
                     "auc_turns_c": round(float(roc_auc_score(low, normal.turns_c[ok])), 3),
                     "auc_centroid": round(float(roc_auc_score(low, normal.centroid[ok])), 3),
                     "spearman_turns_c": round(float(normal.turns_c[ok].rank().corr(x[ok].rank())), 3),
                     "low_share_top40": round(float((top.head(40)[col] <= 6).mean()), 3),
                     "low_share_top100": round(float((top.head(100)[col] <= 6).mean()), 3)})
    nat = s[s.kind.isin(["natural", "normal"])]
    rows.append({"grade": "естественно невежливые (≤ 2) против нормальных", "low_share": round(float((nat.kind == "natural").mean()), 3),
                 "auc_turns_c": round(float(roc_auc_score(nat.kind == "natural", nat.turns_c)), 3),
                 "auc_centroid": round(float(roc_auc_score(nat.kind == "natural", nat.centroid)), 3)})
    g = pd.DataFrame(rows)
    g.to_csv(RESULTS / "calls_grades.csv", index=False)
    print(g.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
