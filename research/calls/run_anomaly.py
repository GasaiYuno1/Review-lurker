#!/usr/bin/env python3
"""Задача 2: поиск аномальных звонков внутри тем. Для каждой модели эмбеддингов и варианта текста:
  1) эмбеддинги звонков (весь диалог) и реплик оператора (кэш build/research/calls_emb/);
  2) темы: UMAP → HDBSCAN по эмбеддингам звонков, выбросы к ближайшей теме (lurker.cluster + anomaly.assign_topics);
  3) оценки аномальности внутри темы: centroid (схема ТЗ), knn, lof, turns (по репликам), turns_c (по центрированным
     репликам оператора), client_c (то же по репликам клиента), offtopic (реплики ближе к чужой теме — под подмену
     куска разговора), coherence, relevance, комбинации;
  4) метрики против списков из prepare_calls.py: все аномалии и по типам (splice / rude / absurd), доля испорченных
     (шум STT) в топ-K — ложные тревоги; естественно невежливые (вежливость ≤ 2) — отдельно.
Варианты текста: stt — все диалоги в стиле распознавания речи (нижний регистр, без пунктуации; так выглядят реальные
расшифровки, и испорченные не выделяются одним стилем); raw — как в датасете.
→ research/results/calls_metrics.csv, research/results/calls_thresholds.csv, build/research/calls_scores.parquet
python3 research/calls/run_anomaly.py [--models tfidf,rubert-tiny2,e5-small] [--variants stt,raw]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BUILD, DATA, RESULTS  # noqa: E402
from lurker.anomaly import (assign_topics, center, centroid_score, coherence_score, knn_score, lof_score, offtopic_score, ranking_metrics,  # noqa: E402
                            relevance_score, robust_z, turn_score)
from lurker.cluster import cluster_embeddings  # noqa: E402
from lurker.embed import get_embedder  # noqa: E402

EMB = BUILD / "calls_emb"
KS = (40, 80, 120, 200)


def stt(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", text.lower())).strip()


def texts_for(df: pd.DataFrame, variant: str) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Тексты звонков и ВСЕХ реплик по порядку (владелец и говорящий — для оценок по репликам)."""
    docs, turns, owner, speaker = [], [], [], []
    norm = stt if variant == "stt" else (lambda x: x)
    for i, tj in enumerate(df["turns"]):
        t = json.loads(tj)
        docs.append("\n".join(("оператор: " if s == "op" else "клиент: ") + norm(x) for s, x in t))
        for s, x in t:
            turns.append(norm(x) or "-")
            owner.append(i)
            speaker.append(s)
    return docs, turns, np.array(owner), np.array(speaker)


def embed_cached(model: str, variant: str, what: str, texts: list[str]) -> np.ndarray:
    p = EMB / f"{variant}__{what}__{model}.npy"
    if p.exists():
        return np.load(p)
    t = time.time()
    e = get_embedder(model).fit(texts)
    E = e.encode(texts).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, E)
    print(f"  эмбеддинги {what} {model}/{variant}: {E.shape}, {time.time() - t:.0f} с", flush=True)
    return E


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf,rubert-tiny2,e5-small,user-base")
    ap.add_argument("--variants", default="stt,raw")
    a = ap.parse_args()
    df = pd.read_parquet(DATA / "calls" / "calls.parquet")
    kind = df["kind"].to_numpy()
    anomaly = kind == "anomaly"
    corrupted = kind == "corrupted"
    types = df["anomaly_type"].to_numpy()
    natural = kind == "natural"
    pool_inj = ~natural                      # подмешанные аномалии: пул без естественных (они не «нормальные»)
    pool_nat = kind != "anomaly"             # естественные: пул без подмешанных
    out_p = RESULTS / "calls_metrics.csv"
    rows = pd.read_csv(out_p).to_dict("records") if out_p.exists() else []
    scores_all = {}
    for variant in a.variants.split(","):
        docs, turns, owner, speaker = texts_for(df, variant)
        op = speaker == "op"
        for model in a.models.split(","):
            print(f"== {model} / {variant}", flush=True)
            E = embed_cached(model, variant, "dialog", docs)
            T = embed_cached(model, variant, "allturns", turns)
            t = time.time()
            res = cluster_embeddings(E, docs, method="umap_hdbscan", outlier_threshold=-1.0)
            topics = assign_topics(E, res.labels)
            n_topics = len(np.unique(topics))
            # доля испорченных, собравшихся в «свою» тему (признак того, что стиль важнее смысла)
            corrupt_topic = max(((topics == tp) & corrupted).sum() / max((topics == tp).sum(), 1) for tp in np.unique(topics))
            n = len(df)
            raw_scores = {"centroid": centroid_score(E, topics), "knn": knn_score(E, topics), "lof": lof_score(E, topics),
                          "turns": turn_score(T[op], owner[op], topics, k=10, top=1),
                          "coherence": coherence_score(T, owner, n), "relevance": relevance_score(T, owner, speaker, n)}
            # центрирование реплик: вычесть общий «жанровый» компонент и нормировать заново (см. extra.py)
            Tc = center(T)
            cl = speaker == "cl"
            raw_scores["turns_c"] = turn_score(Tc[op], owner[op], topics, k=10, top=1)
            # реплики клиента: подмешанный кусок чужого диалога содержит и реплики клиента из другой области
            raw_scores["client_c"] = turn_score(Tc[cl], owner[cl], topics, k=10, top=1)
            # «чужая тема по репликам»: реплика ближе к центру другой темы, чем к своей (подмена куска разговора)
            raw_scores["offtopic"] = offtopic_score(Tc, owner, topics, top=3)
            z = {k: robust_z(v, topics) for k, v in raw_scores.items()}
            z["op+client"] = np.maximum(z["turns_c"], z["client_c"])           # звонок подозрителен, если чужая реплика у любого
            z["op+client_m1"] = np.maximum(z["turns_c"], z["client_c"] - 1.0)  # клиент говорит свободнее: его новизне нужен запас
            z["op+offtopic"] = np.maximum(z["turns_c"], z["offtopic"] - 1.0)
            z["knn+turns"] = np.maximum(z["knn"], z["turns"])
            z["turns+coherence"] = np.maximum(z["turns"], z["coherence"])
            z["max3"] = np.max([z["turns"], z["coherence"], z["relevance"]], axis=0)
            z["rank_mean"] = np.mean([pd.Series(v).rank(pct=True).to_numpy() for v in (z["centroid"], z["knn"], z["lof"], z["turns"])], axis=0)
            z["rank_mean6"] = np.mean([pd.Series(z[k]).rank(pct=True).to_numpy() for k in raw_scores if k not in ("turns_c", "client_c", "offtopic")], axis=0)
            print(f"  тем {n_topics}, кластеризация+оценки {time.time() - t:.0f} с; макс. доля испорченных в одной теме {corrupt_topic:.2f}", flush=True)
            for method, s in z.items():
                base = {"model": model, "variant": variant, "method": method, "n_topics": n_topics,
                        "max_corrupt_topic_share": round(float(corrupt_topic), 3)}
                m = ranking_metrics(s[pool_inj], anomaly[pool_inj], KS, corrupted[pool_inj])
                row = dict(base, target="all", **{k: round(v, 4) for k, v in m.items()})
                for ty in ("splice", "splice_near", "rude", "absurd"):
                    keep = pool_inj & ((types == ty) | ~anomaly)            # пул: этот тип против неаномальных
                    mt = ranking_metrics(s[keep], (types == ty)[keep], (40,))
                    row[f"ap_{ty}"], row[f"auc_{ty}"], row[f"p@40_{ty}"] = round(mt["ap"], 4), round(mt["auc"], 4), round(mt["p@40"], 4)
                mn = ranking_metrics(s[pool_nat], natural[pool_nat], (50,))
                row["ap_natural"], row["auc_natural"], row["p@50_natural"] = round(mn["ap"], 4), round(mn["auc"], 4), round(mn["p@50"], 4)
                rows = [r for r in rows if (r["model"], r["variant"], r["method"], r.get("target")) != (model, variant, method, "all")] + [row]
                scores_all[f"{variant}__{model}__{method}"] = s
            best = max((r for r in rows if r["model"] == model and r["variant"] == variant), key=lambda r: r["ap"])
            print(f"  лучшее: {best['method']} AP={best['ap']:.3f} AUC={best['auc']:.3f} P@120={best['p@120']:.2f} "
                  f"R@120={best['r@120']:.2f} шум@120={best['noise@120']:.2f} | AP по типам splice {best['ap_splice']:.2f} "
                  f"rude {best['ap_rude']:.2f} absurd {best['ap_absurd']:.2f} | естественные AP {best['ap_natural']:.2f}", flush=True)
            pd.DataFrame(rows).to_csv(out_p, index=False)
            sc = pd.DataFrame({"id": df["id"], "topic": topics, **{k: v for k, v in z.items()}})
            sc.to_parquet(BUILD / f"calls_scores__{variant}__{model}.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
