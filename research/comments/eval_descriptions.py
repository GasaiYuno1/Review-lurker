#!/usr/bin/env python3
"""Задача 1, проверка описаний кластеров там, где есть настоящие метки (geo — рубрика, tickets — скрытая тема).
Для каждого описанного кластера: главная метка и её доля (чистота), доля «чужих» текстов в выборке, которую видела LLM,
и насколько её список «выбивающихся» совпадает с чужими по меткам (точность/полнота самооценки LLM).
Плюс для отзывов о банках (меток темы нет, есть банк и рейтинг): какой банк преобладает в кластере и тональность кластера
— видно, какие кластеры собраны по бренду, и «карта боли».
python3 research/comments/eval_descriptions.py → research/results/descriptions_eval.csv, bank_clusters_brand_rating.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BUILD, DATA, RESULTS  # noqa: E402


def main() -> int:
    rows = []
    for ds in ("geo", "tickets"):
        p = RESULTS / f"clusters_{ds}.json"
        if not p.exists():
            continue
        labels = pd.read_parquet(DATA / "comments" / f"{ds}.parquet")["label"].astype(str).to_numpy()
        for c in json.loads(p.read_text(encoding="utf-8"))["clusters"]:
            docs = c.get("sample_docs")
            if not docs or "name" not in c:
                continue
            alien = np.array([labels[i] != c["top_label"] for i in docs])      # «чужие» по настоящей метке
            flagged = np.zeros(len(docs), bool)
            flagged[[k - 1 for k in c.get("outliers", [])]] = True
            tp = int((alien & flagged).sum())
            rows.append({"dataset": ds, "cluster": c["id"], "size": c["size"], "name": c["name"], "top_label": c["top_label"],
                         "purity": c["top_label_share"], "homogeneity_llm": c.get("homogeneity", ""),
                         "sample": len(docs), "alien": int(alien.sum()), "flagged": int(flagged.sum()), "hit": tp,
                         "flag_precision": round(tp / flagged.sum(), 2) if flagged.any() else np.nan,
                         "flag_recall": round(tp / alien.sum(), 2) if alien.any() else np.nan,
                         "alien_random_half": int(alien[len(docs) // 2:].sum()),          # вторая половина — случайные тексты
                         "flagged_random_half": int(flagged[len(docs) // 2:].sum()),
                         "hit_random_half": int((alien & flagged)[len(docs) // 2:].sum())})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "descriptions_eval.csv", index=False)
    for ds, g in df.groupby("dataset"):
        tp, fl, al = g.hit.sum(), g.flagged.sum(), g.alien.sum()
        print(f"{ds}: кластеров {len(g)}, средняя чистота {g.purity.mean():.2f} (взвеш. {np.average(g.purity, weights=g['size']):.2f}); "
              f"«однородность» от LLM: {g.homogeneity_llm.value_counts().to_dict()}")
        print(f"   чужих в выборках {al}/{g['sample'].sum()}, LLM отметила {fl}, из них чужих {tp} "
              f"(точность {tp / max(fl, 1):.2f}, полнота {tp / max(al, 1):.2f})")
        half = g["sample"].sum() / 2
        print(f"   в случайной половине выборки: отмечено {g.flagged_random_half.sum()}/{fl}, из них чужих "
              f"{g.hit_random_half.sum() / max(g.flagged_random_half.sum(), 1):.2f} при доле чужих в этой половине "
              f"{g.alien_random_half.sum() / half:.2f}")
        print(f"   корреляция «доля отмеченных» и «доля чужих» по кластерам: {np.corrcoef(g.flagged / g['sample'], g.alien / g['sample'])[0, 1]:.2f}")
    bank_brands()
    return 0


def bank_brands() -> None:
    p = RESULTS / "clusters_bank.json"
    if not p.exists():
        return
    meta = json.loads(p.read_text(encoding="utf-8"))
    lab_path = BUILD / "clusters" / f"{meta['key']}.npy"
    if not lab_path.exists():
        return
    b = pd.read_parquet(DATA / "comments" / "bank.parquet")
    lab = np.load(lab_path)
    rows = []
    for c in sorted(meta["clusters"], key=lambda c: -c["size"]):
        m = lab == c["id"]
        vc = b["source"][m].value_counts(normalize=True)
        rows.append({"cluster": c["id"], "size": int(m.sum()), "name": c.get("name", "")[:45], "top_bank": vc.index[0],
                     "top_bank_share": round(float(vc.iloc[0]), 2), "bank_share_in_corpus": round(float((b["source"] == vc.index[0]).mean()), 3),
                     "rating_mean": round(float(b["rating"][m].mean()), 2), "neg_share": round(float((b["rating"][m] <= 2).mean()), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "bank_clusters_brand_rating.csv", index=False)
    print(f"банки: кластеров ≥ 90% одного банка — {(out.top_bank_share >= 0.9).sum()} из {len(out)}; "
          f"доля негатива в корпусе {(b['rating'] <= 2).mean():.2f}")


if __name__ == "__main__":
    sys.exit(main())
