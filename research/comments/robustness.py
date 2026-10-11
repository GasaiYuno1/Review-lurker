#!/usr/bin/env python3
"""Задача 1, проверка устойчивости выводов (аудит):
  dirt  — какой вклад в потерю качества даёт мусор из ТЗ сам по себе и как очистка справляется с мусором, под который
          её правила не писались:
            tzdirty / tzclean     — в КАЖДОМ отзыве только ссылка, имя и телефон (ровно просьба ТЗ), без SMM-формата;
            altdirty / altclean   — те же виды мусора в других форматах + 40% текстов в виде выгрузки CRM
                                    (lurker.dirty, уровень alt); очистка — правила как есть;
            altclean_bp           — то же, но перед правилами корпусное удаление шаблонов (lurker.clean.remove_boilerplate);
            clean_bp              — основной «замусоренный» вариант (dirty) с удалением шаблонов: не вредит ли оно;
          метрики как в experiment.py + сохранность слов исходного текста, доля чужих слов и остатки ПД после очистки;
  seeds — разброс качества от зерна UMAP (5 зёрен) для главных моделей: различия между моделями — сигнал или шум.
python3 research/comments/robustness.py [prepare] [embed] [dirt] [seeds]   (без аргументов — всё по порядку)
→ data/research/comments/{geo,tickets}_robust.parquet, research/results/robust_dirt.csv, research/results/seeds.csv
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, DATA, RESULTS  # noqa: E402
from embed import emb_path  # noqa: E402
from lurker.clean import RuleCleaner, remove_boilerplate  # noqa: E402
from lurker.cluster import cluster_embeddings, evaluate, knn_label_share, knn_overlap  # noqa: E402
from lurker.dirty import make_dirty  # noqa: E402
from lurker.embed import get_embedder  # noqa: E402

NEW = {"geo": ["tzdirty", "tzclean", "altdirty", "altclean", "altclean_bp", "clean_bp"], "tickets": ["clean_bp"]}
MODELS = ["tfidf", "hybrid", "e5-small", "user-base", "user-bge-m3"]
HEAVY_ONLY = {"user-bge-m3": {("tickets", "clean_bp")}}   # bge-m3 на CPU ~30 мин на вариант отзывов: только короткие тикеты
SEED_MODELS = ["hybrid", "e5-small", "e5-base", "user-base", "user-bge-m3"]
SEED_PLAN = [("geo", "orig"), ("geo", "clean"), ("tickets", "clean")]
SEED_PLAN_BP = [("tickets", "clean_bp")]                     # удаление шаблонов на тикетах: прирост — сигнал или шум
SEEDS = [42, 1, 2, 3, 4]
WORD_RE = re.compile(r"[а-яёa-z]{2,}")


def robust_path(ds: str) -> Path:
    return DATA / "comments" / f"{ds}_robust.parquet"


def prepare() -> None:
    cleaner = RuleCleaner()
    g = pd.read_parquet(DATA / "comments" / "geo.parquet")
    tz = make_dirty(g["orig"].tolist(), "tz", seed=11)
    alt = make_dirty(g["orig"].tolist(), "alt", seed=12, smm_share=0.4)
    g["tzdirty"] = [d.text for d in tz]
    g["tz_added"] = [json.dumps(d.added, ensure_ascii=False) for d in tz]
    g["tzclean"] = [cleaner.clean(t).text for t in g["tzdirty"]]
    g["altdirty"] = [d.text for d in alt]
    g["alt_kind"] = [d.kind for d in alt]
    g["alt_added"] = [json.dumps(d.added, ensure_ascii=False) for d in alt]
    g["altclean"] = [cleaner.clean(t).text for t in g["altdirty"]]
    bp, removed = remove_boilerplate(g["altdirty"].tolist())
    g["altclean_bp"] = [cleaner.clean(t).text for t in bp]
    bp2, removed2 = remove_boilerplate(g["dirty"].tolist())
    g["clean_bp"] = [cleaner.clean(t).text for t in bp2]
    g.to_parquet(robust_path("geo"))
    t = pd.read_parquet(DATA / "comments" / "tickets.parquet")
    bp3, removed3 = remove_boilerplate(t["dirty"].tolist())
    t["clean_bp"] = [cleaner.clean(x).text for x in bp3]
    t.to_parquet(robust_path("tickets"))
    (BUILD / "boilerplate_removed.json").write_text(json.dumps(
        {"geo_alt": removed, "geo_mix": removed2, "tickets": removed3}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"шаблонов удалено: geo alt {len(removed)}, geo mix {len(removed2)}, tickets {len(removed3)}", flush=True)


def embed() -> None:
    for model in MODELS:
        emb = None
        for ds, variants in NEW.items():
            df = pd.read_parquet(robust_path(ds))
            for v in variants:
                if model in HEAVY_ONLY and (ds, v) not in HEAVY_ONLY[model]:
                    continue
                p = emb_path(ds, v, model)
                if p.exists():
                    continue
                texts = [x if x and x.strip() else "-" for x in df[v].fillna("").astype(str)]
                t = time.time()
                if model in ("tfidf", "hybrid"):
                    E = get_embedder(model).fit(texts).encode(texts)
                else:
                    emb = emb or get_embedder(model)
                    E = emb.encode(texts)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, np.asarray(E, dtype=np.float32))
                print(f"{ds} {v} {model}: {time.time() - t:.0f} с", flush=True)


def clean_stats(orig: list[str], clean: list[str], added: list[str]) -> dict:
    """Сохранность слов исходного текста, доля чужих слов в очищенном, остатки подмешанных ПД."""
    keep, junk, left = [], [], Counter()
    for o, c, a in zip(orig, clean, added):
        wo, wc = Counter(WORD_RE.findall(o.lower())), Counter(WORD_RE.findall(c.lower()))
        common = sum((wo & wc).values())
        keep.append(common / max(sum(wo.values()), 1))
        junk.append((sum(wc.values()) - common) / max(sum(wc.values()), 1))
        digits_c = re.sub(r"\D", "", c)
        for k, v in json.loads(a).items():
            if k == "phone":
                d = re.sub(r"\D", "", v)
                left[k] += bool(d) and d[-7:] in digits_c
            elif k in ("link", "email"):
                core = re.sub(r"^(?:https?://)?(?:www\.)?", "", v).split("/")[0].split("@")[0]
                left[k] += core.lower() in c.lower()
            elif k == "name":
                toks = [w for w in WORD_RE.findall(v.lower()) if w not in o.lower() and len(w) > 2 and w != "ув"]
                left[k] += bool(toks) and any(w in c.lower() for w in toks)
    n = len(orig)
    out = {"retention": round(float(np.mean(keep)), 4), "junk_share": round(float(np.mean(junk)), 4)}
    out.update({f"left_{k}": round(v / n, 4) for k, v in left.items()})
    return out


def dirt() -> None:
    rows = []
    g = pd.read_parquet(robust_path("geo"))
    orig_lab = {}
    for model in MODELS:
        p = BUILD / "clusters" / f"geo__orig__{model}.npy"
        if p.exists():
            orig_lab[model] = np.load(p)
    kinds = {"dirty": (g["dirty_kind"].to_numpy(), "smm"), "clean": (g["dirty_kind"].to_numpy(), "smm"),
             "clean_bp": (g["dirty_kind"].to_numpy(), "smm"), "altdirty": (g["alt_kind"].to_numpy(), "crm"),
             "altclean": (g["alt_kind"].to_numpy(), "crm"), "altclean_bp": (g["alt_kind"].to_numpy(), "crm")}
    stats = {"tzclean": clean_stats(g["orig"].tolist(), g["tzclean"].tolist(), g["tz_added"].tolist()),
             "altclean": clean_stats(g["orig"].tolist(), g["altclean"].tolist(), g["alt_added"].tolist()),
             "altclean_bp": clean_stats(g["orig"].tolist(), g["altclean_bp"].tolist(), g["alt_added"].tolist())}
    for ds, variants in [("geo", ["orig", "dirty", "clean", "tzdirty", "tzclean", "altdirty", "altclean", "altclean_bp", "clean_bp"]),
                         ("tickets", ["dirty", "clean", "clean_bp"])]:
        df = g if ds == "geo" else pd.read_parquet(robust_path("tickets"))
        for model in MODELS:
            Eo = np.load(emb_path(ds, "orig", model)) if ds == "geo" and emb_path(ds, "orig", model).exists() else None
            for v in variants:
                p = emb_path(ds, v, model)
                if not p.exists():
                    continue
                E = np.load(p)
                texts = df[v].fillna("").astype(str).tolist()
                res = cluster_embeddings(E, texts, method="umap_hdbscan")
                lab = res.labels
                row = {"dataset": ds, "model": model, "variant": v, "n_clusters": len(res.topics)}
                row.update({k: round(x, 4) for k, x in evaluate(lab, df["label"].tolist()).items() if k not in ("n_clusters",)})
                if Eo is not None and v != "orig":
                    row["knn_overlap"] = round(knn_overlap(E, Eo), 4)
                    if model in orig_lab:
                        from sklearn.metrics import adjusted_rand_score
                        row["ari_vs_orig"] = round(float(adjusted_rand_score(orig_lab[model], lab)), 4)
                if ds == "geo" and v in kinds:
                    kind, target = kinds[v]
                    row["fmt_share"] = round(float((kind == target).mean()), 4)
                    row["fmt_nn_share"] = round(knn_label_share(E, kind, target), 4)
                if ds == "geo":
                    row.update(stats.get(v, {}))
                rows.append(row)
                pd.DataFrame(rows).to_csv(RESULTS / "robust_dirt.csv", index=False)
                print({k: row[k] for k in ("dataset", "model", "variant", "nmi", "bcubed_p") if k in row}, flush=True)


def seeds(plan=None, models=None) -> None:
    out = RESULTS / "seeds.csv"
    rows = [] if plan is None or not out.exists() else pd.read_csv(out).to_dict("records")
    for ds, v in plan or SEED_PLAN:
        df = pd.read_parquet(DATA / "comments" / f"{ds}.parquet")
        if v not in df:
            df = pd.read_parquet(robust_path(ds))
        rows = [r for r in rows if (r["dataset"], r["variant"]) != (ds, v)]
        texts = df[v].fillna("").astype(str).tolist()
        for model in models or SEED_MODELS:
            p = emb_path(ds, v, model)
            if not p.exists():
                continue
            E = np.load(p)
            for seed in SEEDS:
                res = cluster_embeddings(E, texts, method="umap_hdbscan", random_state=seed)
                ev = evaluate(res.labels, df["label"].tolist())
                rows.append({"dataset": ds, "variant": v, "model": model, "seed": seed, "n_clusters": len(res.topics),
                             **{k: round(ev[k], 4) for k in ("nmi", "bcubed_p", "bcubed_r", "purity", "clean_clusters")}})
                pd.DataFrame(rows).to_csv(RESULTS / "seeds.csv", index=False)
            r = pd.DataFrame(rows)
            r = r[(r.dataset == ds) & (r.variant == v) & (r.model == model)]
            print(f"{ds} {v} {model}: NMI {r.nmi.mean():.3f} ± {r.nmi.std():.3f}", flush=True)


def main() -> int:
    steps = sys.argv[1:] or ["prepare", "embed", "dirt", "seeds"]
    for s in steps:
        {"prepare": prepare, "embed": embed, "dirt": dirt, "seeds": seeds,
         "seeds_bp": lambda: seeds(SEED_PLAN_BP, ["user-base", "user-bge-m3", "e5-base"])}[s]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
