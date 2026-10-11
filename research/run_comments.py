#!/usr/bin/env python3
"""Задача 1 на своих данных: комментарии → удаление шаблонов по частоте в корпусе → очистка правилами и NER → эмбеддинги → UMAP + HDBSCAN → ключевые слова → (по желанию)
название и описание каждого кластера локальной LLM по 20 текстам → таблицы для аналитика.
Настройки — рекомендованные по итогам исследования (research/REPORT.md).

python3 research/run_comments.py tickets.xlsx --text "Комментарий" [--date "Дата"] [--model user-bge-m3] [--mcs 0]
        [--llm configs/llamacpp.yaml] [--describe 30] [--out build/run_comments] [--no-clean]
        [--split-large 0.15] [--merge 0.8] [--mask configs/mask_banks.txt] [--sentiment]
Крупные кластеры (> 15% корпуса — обычно «мешок» благодарностей или общих жалоб) делятся второй ступенью. --mask
удаляет названия брендов до эмбеддингов: без этого отзывы об одном банке/продукте собираются в свой
кластер независимо от темы (research/comments/bank_brands.py).

Выход (в --out):
  clusters.csv     — кластеры: размер, доля, ключевые слова, название и описание LLM, 3 типичных текста;
  assignments.csv  — исходная таблица + номер и название кластера + очищенный текст;
  manual_check.csv — по 10 случайных текстов каждого кластера и пустая колонка «подходит» для ручной проверки;
  dynamics.csv     — (если задан --date) число комментариев каждого кластера по месяцам;
  new_topics.csv   — (если задан --date) кандидаты в новые темы по месяцам: группы комментариев, не похожих на прошлые;
  unusual.csv      — самые необычные комментарии (непохожи на все остальные) — список на прочтение;
  clusters.md      — то же читаемым отчётом;
  boilerplate.csv  — что удалено как шаблон (фразы и служебные метки, повторяющиеся во многих текстах) — проверить глазами.
Реальные данные — только в локальную LLM (llama.cpp/Ollama) или корпоративный endpoint, не во внешние API.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lurker.clean import clean_texts, remove_boilerplate  # noqa: E402
from lurker.cluster import cluster_embeddings, describe_clusters, reduce_topics, split_large  # noqa: E402
from lurker.data import load_table, parse_dates  # noqa: E402
from lurker.embed import get_embedder  # noqa: E402


def embed(texts: list[str], model: str, cache_dir: Path) -> np.ndarray:
    """Эмбеддинги с кэшем на диске (повторный запуск с другими настройками кластеризации — без пересчёта)."""
    h = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    p = cache_dir / f"emb_{model}_{h}.npy"
    if p.exists():
        return np.load(p)
    t = time.time()
    E = get_embedder(model).fit(texts).encode(texts).astype(np.float32)
    np.save(p, E)
    print(f"эмбеддинги {model}: {E.shape[0]} текстов за {time.time() - t:.0f} с", flush=True)
    return E


def llm_sample(E: np.ndarray, idx: np.ndarray, n: int, rng: np.random.Generator) -> list[int]:
    """Половина — ближе всего к центру кластера, половина — случайные (честнее для описания всего кластера)."""
    c = E[idx].mean(axis=0)
    near = idx[np.argsort(-(E[idx] @ c))][: n // 2]
    rest = np.setdiff1d(idx, near)
    rnd = rng.choice(rest, min(len(rest), n - len(near)), replace=False) if len(rest) else []
    return [int(i) for i in list(near) + list(rnd)]


def main(argv=None, llm=None) -> int:
    """llm — готовый клиент (для тестов); иначе строится из --llm."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="CSV или XLSX")
    ap.add_argument("--text", required=True, help="колонка с комментарием")
    ap.add_argument("--date", help="колонка с датой (для динамики по месяцам)")
    ap.add_argument("--model", default="user-bge-m3",
                    help="user-bge-m3 — короткие тикеты (по умолчанию); user-base — длинные отзывы, втрое быстрее; hybrid — мгновенно")
    ap.add_argument("--mcs", type=int, default=0, help="min_cluster_size HDBSCAN (0 — авто: √n/2)")
    ap.add_argument("--merge", type=float, default=0.0,
                    help="сливать кластеры со сходством центров ≥ порога (0 — не сливать; на 20–40 тыс. текстов — 0.8: "
                         "темы дробятся на осколки, research/results/scale_40k_settings.csv; на 3 тыс. слияние вредит)")
    ap.add_argument("--split-large", type=float, default=0.15,
                    help="кластеры больше этой доли корпуса делить второй ступенью (0 — не делить)")
    ap.add_argument("--mask", help="файл с регулярными выражениями (по одному в строке): совпадения удаляются из текста "
                                   "перед эмбеддингами — чтобы названия банков/продуктов не собирали кластеры сами по себе")
    ap.add_argument("--llm", help="конфиг LLM для названий кластеров, например configs/llamacpp.yaml")
    ap.add_argument("--describe", type=int, default=30, help="сколько крупнейших кластеров описывать LLM")
    ap.add_argument("--n-texts", type=int, default=20, help="текстов кластера в запрос к LLM")
    ap.add_argument("--no-clean", action="store_true", help="не чистить текст (только для сравнения)")
    ap.add_argument("--sentiment", nargs="?", const="blanchefort/rubert-base-cased-sentiment",
                    help="тональность каждого комментария готовой моделью → доли негатива/позитива по кластерам "
                         "(работает на словах клиентов; на заметках операторов ошибается — см. research/REPORT.md)")
    ap.add_argument("--no-boilerplate", action="store_true", help="не удалять шаблоны по частоте в корпусе")
    ap.add_argument("--out", default=str(ROOT / "build" / "run_comments"))
    ap.add_argument("--unusual", type=int, default=50, help="сколько самых необычных комментариев выписать в unusual.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    df = load_table(a.file)
    if a.text not in df.columns:
        raise SystemExit(f"нет колонки «{a.text}»; есть: {', '.join(df.columns)}")
    raw = df[a.text].fillna("").astype(str).tolist()
    t = time.time()
    if a.no_clean:
        texts = raw
    else:
        pre = raw
        if not a.no_boilerplate:
            pre, removed = remove_boilerplate(raw)
            pd.DataFrame(removed, columns=["шаблон", "документов"]).to_csv(out / "boilerplate.csv", index=False)
            print(f"шаблонов удалено: {len(removed)} (список — boilerplate.csv)", flush=True)
        texts = clean_texts(pre)
    if a.mask:
        import re
        pats = [l.strip() for l in Path(a.mask).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        rx = re.compile(r"(?<![\w-])(?:" + "|".join(pats) + r")", re.I)
        texts = [" ".join(rx.sub(" ", x).split()) for x in texts]
        print(f"маскировано названий: {len(pats)} шаблонов", flush=True)
    print(f"{len(texts)} комментариев, очистка {time.time() - t:.0f} с", flush=True)
    ok = np.array([len(x.strip()) >= 3 for x in texts])          # пустые после очистки — вне кластеров
    work = [x for x, k in zip(texts, ok) if k]
    E = embed(work, a.model, out)
    t = time.time()
    res = cluster_embeddings(E, work, method="umap_hdbscan", min_cluster_size=a.mcs or None)
    if a.merge:
        lab_m = reduce_topics(E, res.labels, merge_threshold=a.merge)
        print(f"слияние близких (≥ {a.merge}): кластеров {len(set(res.labels.tolist()) - {-1})} → {len(set(lab_m.tolist()) - {-1})}", flush=True)
        res.labels, res.topics = lab_m, describe_clusters(E, work, lab_m)
    if a.split_large:
        lab2 = split_large(E, work, res.labels, share=a.split_large)
        if len(set(lab2.tolist())) != len(set(res.labels.tolist())):
            res.labels, res.topics = lab2, describe_clusters(E, work, lab2)
            print(f"вторая ступень: крупные кластеры (> {a.split_large:.0%}) разделены", flush=True)
    print(f"кластеров {len(res.topics)}, прочее {res.n_outliers}, кластеризация {time.time() - t:.0f} с", flush=True)
    labels = np.full(len(texts), -1)
    labels[ok] = res.labels
    tone = None
    if a.sentiment:
        from lurker.sentiment import CLASSES, predict_sentiment
        t = time.time()
        tone = np.array(CLASSES)[predict_sentiment(work, a.sentiment).argmax(1)]
        print(f"тональность ({a.sentiment}): {time.time() - t:.0f} с", flush=True)

    if llm is None and a.llm:
        from lurker.config import load_config
        from lurker.llm import LLMClient
        llm = LLMClient(load_config(a.llm).llm, cache_dir=str(ROOT / ".cache"))
    from lurker.summarize import describe_cluster
    rows, check = [], []
    for rank, tp in enumerate(sorted(res.topics, key=lambda x: -x.size)):
        idx = np.where(res.labels == tp.id)[0]
        rec = {"cluster": tp.id, "size": tp.size, "share": round(tp.size / len(texts), 4), "keywords": ", ".join(tp.keywords[:8]),
               "name": "", "description": "", "sentiment": ""}
        if tone is not None:
            rec["negative_share"] = round(float((tone[idx] == "neg").mean()), 3)
            rec["positive_share"] = round(float((tone[idx] == "pos").mean()), 3)
        if llm is not None and rank < a.describe:
            t = time.time()
            try:
                d = describe_cluster([work[i] for i in llm_sample(E, idx, a.n_texts, rng)], llm, tp.keywords)
                rec.update({k: d.get(k, "") for k in ("name", "description", "sentiment")})
            except Exception as e:                                   # сбой LLM не должен ронять весь прогон
                rec["description"] = f"[LLM: {str(e)[:100]}]"
            print(f"  {tp.id:3d} {tp.size:6d}  {rec['name'][:50]:50}  {time.time() - t:.0f} с", flush=True)
        for j, i in enumerate(tp.representatives[:3]):
            rec[f"example_{j + 1}"] = " ".join(work[i].split())[:300]
        rows.append(rec)
        for i in rng.choice(idx, min(10, len(idx)), replace=False):
            check.append({"cluster": tp.id, "name": rec["name"] or rec["keywords"], "text": " ".join(work[i].split())[:600], "подходит": ""})
    cl = pd.DataFrame(rows)
    cl.to_csv(out / "clusters.csv", index=False)
    pd.DataFrame(check).to_csv(out / "manual_check.csv", index=False)

    names = {r["cluster"]: r["name"] or r["keywords"] for r in rows}
    df = df.copy()
    df["cluster"] = labels
    df["cluster_name"] = [names.get(l, "прочее") for l in labels]
    df["clean_text"] = texts
    if tone is not None:
        full = np.full(len(texts), "", dtype=object)
        full[ok] = tone
        df["tone"] = full
    df.to_csv(out / "assignments.csv", index=False)

    md = [f"# Кластеры комментариев: {Path(a.file).name}", "",
          f"{len(texts)} комментариев, модель {a.model}, кластеров {len(rows)}, вне кластеров {int((labels < 0).sum())}.", ""]
    for r in rows:
        md += [f"## {r['name'] or r['keywords']} — {r['size']} ({r['share']:.1%})", "",
               f"Ключевые слова: {r['keywords']}", ""] + ([r["description"], ""] if r["description"] else [])
        if "negative_share" in r:
            md += [f"Тональность: негатив {r['negative_share']:.0%}, позитив {r['positive_share']:.0%}", ""]
        md += [f"> {r[k]}" for k in ("example_1", "example_2", "example_3") if r.get(k)] + [""]
    # необычные комментарии и новые темы (lurker.novelty)
    from lurker.novelty import emerging_groups, unusual_scores
    zu = unusual_scores(E, work)
    o = [i for i in np.argsort(-zu) if np.isfinite(zu[i])][: a.unusual]
    pd.DataFrame({"rank": range(1, len(o) + 1), "z": np.round(zu[o], 2), "cluster_name": [names.get(res.labels[i], "прочее") for i in o],
                  "text": [" ".join(work[i].split())[:600] for i in o]}).to_csv(out / "unusual.csv", index=False)
    md += ["## Самые необычные комментарии", "", "Непохожи на все остальные — список на прочтение (бывает и мусор).", ""]
    md += [f"{r}. {' '.join(work[i].split())[:300]}" for r, i in enumerate(o[:20], 1)] + [""]
    if a.date:
        dw = parse_dates(df[a.date])[ok].to_numpy()
        eg = emerging_groups(E, dw)
        rows_e = [{"period": g.period, "size": g.size, "clusters": "; ".join(pd.Series([names.get(res.labels[i], "прочее") for i in g.members]).value_counts().index[:3]),
                   "examples": " | ".join(" ".join(work[i].split())[:150] for i in g.members[:5])} for g in eg.itertuples()]
        pd.DataFrame(rows_e, columns=["period", "size", "clusters", "examples"]).to_csv(out / "new_topics.csv", index=False)
        md += ["## Кандидаты в новые темы", ""]
        md += [f"- {r['period']}: {r['size']} комментариев — {r['examples'][:300]}" for r in rows_e] or ["Не найдено."]
        md += [""]
        print(f"кандидатов в новые темы: {len(rows_e)} (new_topics.csv)", flush=True)
    if a.date:
        d = parse_dates(df[a.date])
        dyn = (pd.DataFrame({"month": d.dt.to_period("M").astype(str).where(d.notna()), "cluster_name": df["cluster_name"]})
               .dropna().pivot_table(index="cluster_name", columns="month", aggfunc="size", fill_value=0))
        dyn.to_csv(out / "dynamics.csv")
        try:
            table = dyn.to_markdown()
        except ImportError:                                          # to_markdown требует tabulate
            table = "```\n" + dyn.to_string() + "\n```"
        md += ["## Динамика по месяцам", "", table, ""]
    (out / "clusters.md").write_text("\n".join(md), encoding="utf-8")
    print(f"→ {out}/clusters.csv, assignments.csv, manual_check.csv, clusters.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
