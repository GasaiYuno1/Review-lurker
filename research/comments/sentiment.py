#!/usr/bin/env python3
"""Задача 1, «что положительного или отрицательного пишут» (ТЗ): тональность каждого комментария готовой открытой
моделью, без обучения. Проверка там, где ответ известен: отзывы о банках и Яндекс Картах — по рейтингу (1–2 негатив,
3 нейтрально, 4–5 позитив), синтетические тикеты — по скрытой разметке генератора (neg/neu/pos). Тикеты — заметки
оператора о клиенте («абонент недоволен, угрожает…»), а не отзывы: модели, обученные на отзывах, здесь могут ошибаться.
Модели: seara/rubert-tiny2-russian-sentiment (быстрая), seara/rubert-base-cased-russian-sentiment,
blanchefort/rubert-base-cased-sentiment, cointegrated/rubert-tiny-sentiment-balanced.
Метрики: macro-F1 по трём классам; AUC «негатив против остального» по вероятности негатива; на уровне кластеров —
насколько предсказанная доля негатива повторяет настоящую (корреляция, средняя ошибка в процентных пунктах).
python3 research/comments/sentiment.py [модели через запятую] → research/results/sentiment_eval.csv,
  sentiment_clusters_by_model.csv, sentiment_clusters_{tickets,bank}.csv (по лучшей для набора модели);
  предсказания — build/research/sentiment/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BUILD, DATA, RESULTS, ROOT  # noqa: E402
from lurker.sentiment import predict_sentiment  # noqa: E402

MODELS = ["seara/rubert-tiny2-russian-sentiment", "seara/rubert-base-cased-russian-sentiment",
          "blanchefort/rubert-base-cased-sentiment", "cointegrated/rubert-tiny-sentiment-balanced"]
OUT = BUILD / "sentiment"


def predict(model: str, texts: list[str]) -> np.ndarray:
    return predict_sentiment(texts, model)


def truth(ds: str, df: pd.DataFrame) -> np.ndarray:
    if ds == "tickets":
        lab = pd.read_csv(ROOT / "data" / "test" / "tickets_labels.csv")[["ID", "sentiment"]]
        lab["id"] = "t-" + lab["ID"].astype(str)
        return df[["id"]].merge(lab[["id", "sentiment"]], on="id", how="left")["sentiment"].fillna("neu").to_numpy()
    r = df["rating"].to_numpy()
    return np.where(r <= 2, "neg", np.where(r >= 4, "pos", "neu"))


def cluster_table(ds: str, df: pd.DataFrame, y: np.ndarray, P: np.ndarray, labels: np.ndarray, names: dict) -> pd.DataFrame:
    pred = np.array(["neg", "neu", "pos"])[P.argmax(1)]
    rows = []
    for c in sorted(set(labels[labels >= 0].tolist()), key=lambda c: -(labels == c).sum()):
        m = labels == c
        rows.append({"cluster": c, "name": names.get(c, ""), "size": int(m.sum()),
                     "negative_true": round(float((y[m] == "neg").mean()), 3), "negative_pred": round(float((pred[m] == "neg").mean()), 3),
                     "positive_true": round(float((y[m] == "pos").mean()), 3), "positive_pred": round(float((pred[m] == "pos").mean()), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"sentiment_clusters_{ds}.csv", index=False)
    return out


def main() -> int:
    from sklearn.metrics import f1_score, roc_auc_score
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else MODELS
    OUT.mkdir(parents=True, exist_ok=True)
    out_p = RESULTS / "sentiment_eval.csv"
    rows = pd.read_csv(out_p).to_dict("records") if out_p.exists() else []
    for ds in ("tickets", "geo", "bank"):
        df = pd.read_parquet(DATA / "comments" / f"{ds}.parquet")
        keep = df["rating"].to_numpy() > 0 if ds != "tickets" else np.ones(len(df), bool)
        y = truth(ds, df)
        texts = df["clean"].fillna("").astype(str).tolist()
        for model in models:
            p = OUT / f"{ds}__{model.replace('/', '__')}.npy"
            t = time.time()
            if p.exists():
                P = np.load(p)
            else:
                P = predict(model, texts)
                np.save(p, P)
                p.with_suffix(".json").write_text(json.dumps({"texts_per_s": round(len(texts) / (time.time() - t), 1)}))
            speed = json.loads(p.with_suffix(".json").read_text())["texts_per_s"] if p.with_suffix(".json").exists() else None
            pred = np.array(["neg", "neu", "pos"])[P.argmax(1)]
            row = {"dataset": ds, "model": model, "n": int(keep.sum()),
                   "macro_f1": round(float(f1_score(y[keep], pred[keep], average="macro")), 3),
                   "f1_neg": round(float(f1_score(y[keep] == "neg", pred[keep] == "neg")), 3),
                   "f1_pos": round(float(f1_score(y[keep] == "pos", pred[keep] == "pos")), 3),
                   "auc_neg": round(float(roc_auc_score(y[keep] == "neg", P[keep, 0])), 3),
                   "neg_share_true": round(float((y[keep] == "neg").mean()), 3), "neg_share_pred": round(float((pred[keep] == "neg").mean()), 3),
                   "texts_per_s": speed}
            rows = [r for r in rows if (r["dataset"], r["model"]) != (ds, model)] + [row]
            pd.DataFrame(rows).to_csv(out_p, index=False)
            print(row, flush=True)
    # доля негатива по кластерам — то, что нужно заказчику («где ругают»): каждая модель против правды
    agg = []
    for ds, key in (("tickets", "tickets__clean__user-bge-m3"), ("bank", "bank__clean__user-base__mcs20")):
        lab_p = BUILD / "clusters" / f"{key}.npy"
        if not lab_p.exists():
            continue
        df = pd.read_parquet(DATA / "comments" / f"{ds}.parquet")
        y, L = truth(ds, df), np.load(lab_p)
        cs = [c for c in np.unique(L) if c >= 0 and (L == c).sum() >= 30]
        tr = np.array([(y[L == c] == "neg").mean() for c in cs])
        for model in MODELS:
            f = OUT / f"{ds}__{model.replace('/', '__')}.npy"
            if not f.exists():
                continue
            pred = np.array(["neg", "neu", "pos"])[np.load(f).argmax(1)]
            pr = np.array([(pred[L == c] == "neg").mean() for c in cs])
            agg.append({"dataset": ds, "model": model, "кластеров": len(cs), "корреляция доли негатива": round(float(np.corrcoef(tr, pr)[0, 1]), 2),
                        "ошибка, п.п.": round(100 * float(np.abs(tr - pr).mean()))})
        best = max((r for r in agg if r["dataset"] == ds), key=lambda r: r["корреляция доли негатива"])["model"]
        names = {}
        jp = RESULTS / f"clusters_{ds}.json"
        if jp.exists():
            names = {c["id"]: c.get("name", "") for c in json.loads(jp.read_text(encoding="utf-8"))["clusters"]}
        cluster_table(ds, df, y, np.load(OUT / f"{ds}__{best.replace('/', '__')}.npy"), L, names)
        print(f"{ds}: по кластерам лучшая {best}", flush=True)
    ag = pd.DataFrame(agg)
    ag.to_csv(RESULTS / "sentiment_clusters_by_model.csv", index=False)
    print(ag.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
