#!/usr/bin/env python3
"""Оценка пайплайна на тестовых данных со скрытой разметкой (data/test/tickets*.csv):
  1) очистка — остаточные ПД по отделам;
  2) кластеризация — ARI/NMI/чистота против topic для бэкендов эмбеддингов × методов;
  3) динамика — находит ли detect_changes заложенные тренды (по истинным темам и по кластерам).
python3 tools/eval.py [--data data/test/tickets.csv] [--backends tfidf,navec,hybrid] [--methods umap_hdbscan,agglomerative,bertopic] [--quick]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lurker.clean import EMAIL_RE, PHONE_RE, URL_RE, RuleCleaner, clean_texts  # noqa: E402
from lurker.cluster import cluster_embeddings, evaluate  # noqa: E402
from lurker.data import guess_columns, load_table, normalize  # noqa: E402
from lurker.dynamics import detect_changes, timeline  # noqa: E402
from lurker.embed import get_embedder  # noqa: E402

PATR = re.compile(r"[А-ЯЁ][а-яё]+(?:ович|евич|овна|евна)\b")
STREET = re.compile(r"\bул\.|\bд\. ?\d|\bкв\. ?\d|[А-ЯЁ][а-яё]+ \d{1,3}-\d{1,3}")
INITIALS = re.compile(r"[А-ЯЁ][а-яё]+ [А-ЯЁ]\.[А-ЯЁ]\.")


def load(path: str) -> pd.DataFrame:
    raw = load_table(path)
    df = normalize(raw, guess_columns(raw))
    lab = pd.read_csv(Path(path).with_name(Path(path).stem + "_labels.csv"), dtype={"ID": str})
    return df.merge(lab, left_on="id", right_on="ID", how="left")


def eval_clean(df: pd.DataFrame) -> None:
    print("== очистка (доля текстов с остатками)")
    for name, cleaner in (("rules", RuleCleaner(ner=False)), ("rules+ner", RuleCleaner(ner=True))):
        t = time.time()
        df["clean"] = clean_texts(df["text"].tolist(), cleaner)
        dt = time.time() - t
        for dept, sub in df[df["text"].str.len() > 5].groupby("department"):
            c = sub["clean"]
            print(f"  {name:9} {dept:13} n={len(sub):4d} url={c.str.contains(URL_RE).mean():.3f} phone={c.str.contains(PHONE_RE).mean():.3f} "
                  f"email={c.str.contains(EMAIL_RE).mean():.3f} отчество={c.str.contains(PATR).mean():.3f} улица={c.str.contains(STREET).mean():.3f} "
                  f"инициалы={c.str.contains(INITIALS).mean():.3f} пусто={(c == '').mean():.3f}")
        print(f"  {name}: {dt:.1f} с на {len(df)} текстов")


def eval_cluster(df: pd.DataFrame, backends: list[str], methods: list[str]) -> None:
    print("\n== кластеризация против topic (без пустых текстов)")
    m = df["clean"].str.len() > 3
    texts, true = df.loc[m, "clean"].tolist(), df.loc[m, "topic"].tolist()
    print(f"  текстов {len(texts)}, тем в разметке {len(set(true))}")
    best = None
    for backend in backends:
        t = time.time()
        emb = get_embedder(backend).fit(texts)
        E = emb.encode(texts)
        te = time.time() - t
        for method in methods:
            t = time.time()
            try:
                res = cluster_embeddings(E, texts, method=method)
            except Exception as e:
                print(f"  {backend:7} {method:14} ошибка: {type(e).__name__}: {str(e)[:80]}")
                continue
            ev = evaluate(res.labels, true)
            print(f"  {backend:7} {method:14} clusters={ev['n_clusters']:3d} outliers={ev['outliers']:.2f} ARI={ev['ari']:.3f} NMI={ev['nmi']:.3f} "
                  f"purity={ev['purity']:.3f}  emb {te:.1f}s, cluster {time.time() - t:.1f}s")
            if best is None or ev["nmi"] > best[0]:
                best = (ev["nmi"], backend, method, res, E)
    if best:
        _, backend, method, res, E = best
        print(f"\n  лучший: {backend}/{method}; темы (размер, ключевые слова, доминирующая истинная тема):")
        lab = np.array(true)
        for tp in res.topics[:30]:
            idx = np.where(res.labels == tp.id)[0]
            top = pd.Series(lab[idx]).value_counts()
            print(f"   {tp.id:3d} {tp.size:5d}  {', '.join(tp.keywords[:6]):60.60}  {top.index[0]} {top.iloc[0] / tp.size:.0%}")
        df.loc[m, "cluster"] = res.labels
        df["cluster"] = df["cluster"].fillna(-1).astype(int)


def eval_dynamics(df: pd.DataFrame) -> None:
    print("\n== динамика: заложенные тренды")
    planted = df.dropna(subset=["trend"]).groupby("trend")["topic"].unique().to_dict()
    for k, v in planted.items():
        print(f"  {k}: {', '.join(v)}")
    for col in ("topic", "cluster"):
        if col not in df.columns:
            continue
        for freq in ("M", "W"):
            tl = timeline(df, topic_col=col, freq=freq)
            found = detect_changes(tl)
            print(f"\n  по {col}, гранулярность {freq}: {len(found)} находок")
            for f in found[:12]:
                print("   ", f.describe())


QUERY_TRUTH = {
    # промпт → функция: DataFrame с разметкой → маска истины
    "как часто нас оскорбляли и угрожали физической расправой": lambda d: d["threat"] == 1,
    "нас стали чаще хуесосить в последние 2 месяца или нет?": lambda d: d["threat"] == 1,
    "стали ли чаще жаловаться на домофон в приложении?": lambda d: d["topic"].str.startswith("домофон_приложение"),
    "сравни, что говорили про мобильное приложение в январе и в июле": lambda d: d["topic"].str.startswith("прил_") | d["topic"].str.startswith("домофон_приложение"),
    "не приходит звонок с домофона": lambda d: d["topic"] == "домофон_приложение_не_приходит_звонок",
    "двойное списание": lambda d: d["topic"] == "оплата_двойное_списание",
    "хотят расторгнуть договор, уходят к конкурентам": lambda d: d["topic"] == "расторжение_дорого_конкурент",
    "благодарности": lambda d: d["topic"] == "благодарность",
    "низкая скорость интернета по вечерам": lambda d: d["topic"] == "инет_низкая_скорость_вечером",
    "покажи тренд негатива за последние 3 месяца": lambda d: (d["threat"] == 1) | (d["topic"] == "жалоба_на_сотрудника_грубость"),
}


def eval_query(df: pd.DataFrame, thresholds=(0.3, 0.4, 0.5, 0.6)) -> None:
    """Отбор по смыслу из промпта против разметки: точность/полнота по порогам, F1-оптимум, динамика понятия."""
    from lurker.service.concept import build_concept, score
    from lurker.pipeline import run_analysis
    from lurker.service.query import run_query
    print("\n== вопрос → выборка по смыслу (точность p / полнота r по порогам)")
    an = run_analysis(df.drop(columns=[c for c in ("clean", "topic", "topic_name") if c in df.columns]).rename(columns={"topic": "_t"}),
                      cache_dir=ROOT / ".cache")
    d = df.reset_index(drop=True)
    texts = an.df["clean"].tolist()
    ref = an.df["date"].max().date()
    nonempty = (an.df["clean"].str.len() > 3).to_numpy()
    for prompt, fn in QUERY_TRUTH.items():
        c, _ = build_concept(prompt, an.embedder, texts, an.E, ref)
        s = score(c, texts, an.E)
        t = fn(d).to_numpy() & nonempty
        parts, best = [], (0.0, None)
        for thr in thresholds:
            pred = s >= thr
            tp = int((pred & t).sum())
            p, r = tp / max(pred.sum(), 1), tp / max(t.sum(), 1)
            f = 2 * p * r / max(p + r, 1e-9)
            parts.append(f"{thr}: p={p:.2f} r={r:.2f}")
            if f > best[0]:
                best = (f, thr)
        print(f"  {prompt[:50]:50} ист={int(t.sum()):4d} | " + " ".join(parts) + f" | F1={best[0]:.2f}@{best[1]} пресеты={c.presets}")
    for prompt in ("стали ли чаще жаловаться на домофон в приложении?", "как часто нас оскорбляли и угрожали физической расправой"):
        q = run_query(an, prompt, threshold=0.5, freq="M")
        print(f"\n  {prompt}\n  " + q.text.replace("\n", "\n  "))
        print("  по месяцам: " + ", ".join(f"{k:%m}: {int(v)}" for k, v in q.series["count"].items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/test/tickets.csv"))
    ap.add_argument("--backends", default="tfidf,navec,hybrid")
    ap.add_argument("--methods", default="umap_hdbscan,agglomerative,bertopic")
    ap.add_argument("--quick", action="store_true", help="только hybrid/umap_hdbscan")
    ap.add_argument("--query", action="store_true", help="только отбор по смыслу из промптов")
    a = ap.parse_args()
    df = load(a.data)
    if a.query:
        eval_query(df)
        return 0
    eval_clean(df)
    backends = ["hybrid"] if a.quick else a.backends.split(",")
    methods = ["umap_hdbscan"] if a.quick else a.methods.split(",")
    eval_cluster(df, backends, methods)
    eval_dynamics(df)
    eval_query(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
