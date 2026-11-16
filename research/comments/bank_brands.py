#!/usr/bin/env python3
"""Задача 1, отзывы о банках: две доработки «карты боли».
1) Бренды. Часть кластеров собрана по банку (Тинькофф, Альфа, МТС Банк, Газпромбанк), а не по теме. Проверка: заменить
   названия банков и их продуктов в очищенном тексте на слово «банк» (список ниже) и сравнить кластеры: насколько
   кластеры определяются банком (NMI с колонкой «банк», число кластеров ≥ 90% одного банка) и тональностью (NMI с
   классом рейтинга).
2) Вторая ступень. Один кластер благодарностей занимает ~36% отзывов. Крупные кластеры (> 15% корпуса) кластеризуются
   ещё раз внутри себя — получаются подтемы (за что именно хвалят).
Модель — USER-base (длинные отзывы), min_cluster_size = 20, как в описаниях кластеров.
python3 research/comments/bank_brands.py → research/results/bank_brands.csv, bank_subclusters.csv
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, DATA, RESULTS  # noqa: E402
from embed import emb_path  # noqa: E402
from lurker.cluster import class_tfidf_keywords, cluster_embeddings  # noqa: E402
from lurker.embed import get_embedder  # noqa: E402

BRANDS = [r"т[-\s]?банк\w*", r"тинько?фф?\w*", r"тиньков\w*", r"tinkoff\w*", r"сбер(?!еж|ег|еч)\w{0,7}", r"втб\w*", r"альфа[-\s]?банк\w*",
          r"альф(?:а|ы|е|у|ой)\b", r"совком\w*", r"халв\w*", r"газпромбанк\w*", r"гпб\b", r"мтс(?:[-\s]?банк\w*)?", r"озон(?!атор)\w*", r"ozon\w*",
          r"почта[-\s]?банк\w*", r"хоум[-\s]?кредит\w*", r"хоум\w*", r"home\s+credit", r"россельхоз\w*", r"рсхб\w*",
          r"банк\w*\s+[«\"]?открыти\w*[»\"]?", r"[«\"]открыти\w*[»\"]", r"фк\s+открыти\w*", r"райф+айзен\w*", r"уралсиб\w*",
          r"мкб\b", r"московск\w+\s+кредитн\w+\s+банк\w*", r"юникредит\w*", r"росбанк\w*", r"ренессанс\w*", r"модульбанк\w*",
          r"убрир\w*", r"русск\w+\s+стандарт\w*", r"атб\b", r"кредит\s+европа\w*", r"ак\s+барс\w*", r"отп\s*банк\w*", r"отп\b",
          r"юнистрим\w*", r"дом\.?\s?рф", r"промсвязьбанк\w*", r"псб\b", r"еком\w*", r"драйв\s+клик\w*", r"зенит\w*", r"бкс\b",
          r"абсолют\s+банк\w*", r"центр[-\s]инвест\w*", r"ю\s?money", r"юмани\w*", r"киви\b", r"qiwi\w*", r"банки\.ру"]
BRAND_RE = re.compile(r"(?<![\w-])(?:" + "|".join(BRANDS) + r")", re.I)
KEY = "bank__clean__user-base__mcs20"


def rating_class(r) -> np.ndarray:
    return np.where(r <= 2, "neg", np.where(r >= 4, "pos", "neu"))


def describe(lab: np.ndarray, b: pd.DataFrame) -> dict:
    from sklearn.metrics import normalized_mutual_info_score as nmi
    ok = lab >= 0
    sizes = pd.Series(lab[ok]).value_counts()
    brand90 = sum((b["source"][lab == c].value_counts(normalize=True).iloc[0] >= 0.9) for c in sizes.index)
    return {"кластеров": len(sizes), "крупнейший, доля": round(float(sizes.iloc[0] / len(lab)), 3),
            "NMI с банком": round(float(nmi(b["source"][ok], lab[ok])), 3),
            "кластеров ≥ 90% одного банка": int(brand90),
            "NMI с тональностью (рейтинг)": round(float(nmi(rating_class(b["rating"].to_numpy())[ok], lab[ok])), 3)}


def main() -> int:
    b = pd.read_parquet(DATA / "comments" / "bank.parquet")
    clean = b["clean"].fillna("").astype(str)
    nobrand = clean.map(lambda t: BRAND_RE.sub("банк", t))
    print(f"упоминаний брендов заменено: {sum(len(BRAND_RE.findall(t)) for t in clean)}; "
          f"текстов с упоминанием: {int(clean.map(lambda t: bool(BRAND_RE.search(t))).sum())} из {len(b)}", flush=True)
    E = np.load(emb_path("bank", "clean", "user-base"))
    p = emb_path("bank", "clean_nobrand", "user-base")
    if not p.exists():
        t = time.time()
        En = get_embedder("user-base").encode([x if x.strip() else "-" for x in nobrand]).astype(np.float32)
        np.save(p, En)
        print(f"эмбеддинги без брендов: {time.time() - t:.0f} с", flush=True)
    En = np.load(p)
    rows, labs = [], {}
    for name, X, texts in (("очищенный текст", E, clean.tolist()), ("без названий банков", En, nobrand.tolist())):
        res = cluster_embeddings(X, texts, method="umap_hdbscan", min_cluster_size=20)
        labs[name] = (res.labels, X, texts)
        rows.append({"вариант": name, **describe(res.labels, b)})
        print(rows[-1], flush=True)
    np.save(BUILD / "clusters" / f"{KEY}.npy", labs["очищенный текст"][0])                     # для sentiment.py
    np.save(BUILD / "clusters" / "bank__clean_nobrand__user-base__mcs20.npy", labs["без названий банков"][0])
    pd.DataFrame(rows).to_csv(RESULTS / "bank_brands.csv", index=False)

    # вторая ступень: крупные кластеры (> 15% корпуса) — ещё раз внутри себя
    sub = []
    for name, (lab, X, texts) in labs.items():
        two = lab.copy()                                          # двухступенчатая разметка: крупные кластеры заменены подкластерами
        for c in pd.Series(lab[lab >= 0]).value_counts().index:
            idx = np.where(lab == c)[0]
            if len(idx) < 0.15 * len(lab):
                continue
            r = cluster_embeddings(X[idx], [texts[i] for i in idx], method="umap_hdbscan", min_cluster_size=max(15, len(idx) // 40))
            two[idx] = 1000 * (c + 1) + r.labels
            kw = class_tfidf_keywords([texts[i] for i in idx], r.labels, n_keywords=6)
            for s in pd.Series(r.labels).value_counts().index:
                m = r.labels == s
                sub.append({"вариант": name, "кластер": int(c), "размер кластера": len(idx), "подкластер": int(s), "размер": int(m.sum()),
                            "средний рейтинг": round(float(b["rating"].to_numpy()[idx][m].mean()), 2),
                            "ключевые слова": ", ".join(kw.get(s, [])[:6]),
                            "пример": " ".join(texts[idx[np.where(m)[0][0]]].split())[:200]})
        rows.append({"вариант": name + " + вторая ступень", **describe(two, b)})
        np.save(BUILD / "clusters" / f"bank__{'clean' if name == 'очищенный текст' else 'clean_nobrand'}__user-base__two_level.npy", two)
    pd.DataFrame(rows).to_csv(RESULTS / "bank_brands.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    sb = pd.DataFrame(sub)
    sb.to_csv(RESULTS / "bank_subclusters.csv", index=False)
    print(sb.drop(columns=["пример"]).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
