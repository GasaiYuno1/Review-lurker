#!/usr/bin/env python3
"""Задача 1, данные. Три набора, у каждого варианты текста:
  orig  — исходный текст;
  dirty — «замусоренный» (ТЗ: ссылка, имя, телефон в каждый текст; 40% текстов — в формате SMM-тикета
          со ссылкой, перепиской, шаблонными репликами оператора и блоком ПД);
  clean — dirty после очистки правилами + NER (lurker.clean.RuleCleaner).

Наборы:
  bank    — отзывы о банках с banki.ru (Kaggle romanberdyshev/bank-reviews-dataset, датасет из ТЗ): 4000 случайных;
            меток темы нет, есть рейтинг 1–5 и заголовок;
  geo     — отзывы Яндекс Карт 2023 (github yandex/geo-reviews-dataset-2023, зеркало d0rj на HF): 12 непохожих рубрик × 350;
            рубрика — внешняя метка «о чём отзыв», по ней считаются чистота и NMI;
  tickets — наши синтетические тикеты (data/test/tickets.csv, 3000 строк, 25 тем): мусор уже внутри (SMM и пр.),
            поэтому orig нет, dirty = как есть, clean = очистка; метка — скрытая тема.
python3 research/comments/prepare_data.py [--n-bank 4000] [--per-rubric 350] [--seed 1]
→ data/research/comments/{bank,geo,tickets}.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RAW, ROOT, download, unzip_one  # noqa: E402
from lurker.clean import RuleCleaner  # noqa: E402
from lurker.dirty import make_dirty  # noqa: E402

BANK_URL = "https://www.kaggle.com/api/v1/datasets/download/romanberdyshev/bank-reviews-dataset"
GEO_URL = "https://huggingface.co/datasets/d0rj/geo-reviews-dataset-2023/resolve/main/data/train-00000-of-00001-49261e4e5a35a5a0.parquet"
RUBRICS = ["Гостиница", "Ресторан", "Супермаркет", "Музей", "Пункт выдачи", "Аптека", "Стоматологическая клиника",
           "Салон красоты", "АЗС", "Автосервис, автотехцентр", "Банк", "Детский сад, ясли"]
OUT = DATA / "comments"


def add_variants(df: pd.DataFrame, seed: int, smm_share: float = 0.4, cleaner: RuleCleaner | None = None) -> pd.DataFrame:
    cleaner = cleaner or RuleCleaner()
    dirty = make_dirty(df["orig"].tolist(), "mix", seed=seed, smm_share=smm_share)
    df = df.copy()
    df["dirty"] = [d.text for d in dirty]
    df["dirty_kind"] = [d.kind for d in dirty]
    df["clean"] = [cleaner.clean(t).text for t in df["dirty"]]
    return df


def bank(n: int, seed: int) -> pd.DataFrame:
    z = download(BANK_URL, RAW / "bank-reviews-dataset.zip")
    csv = unzip_one(z, "final_review_dataset_extended.csv")
    b = pd.read_csv(csv)
    b = b[b["review"].astype(str).str.len() > 50].sample(n, random_state=seed).reset_index(drop=True)
    return pd.DataFrame({"id": [f"bank-{i}" for i in range(len(b))], "orig": b["review"].astype(str).str.strip(),
                         "title": b["review_title"].astype(str), "rating": b["rating_value"].astype(int),
                         "source": b["bank"].astype(str), "date": b["review_dttm"].astype(str), "label": ""})


def geo(per_rubric: int, seed: int) -> pd.DataFrame:
    p = download(GEO_URL, RAW / "geo-reviews-2023.parquet")
    g = pd.read_parquet(p)
    g = g[g["rubrics"].isin(RUBRICS) & (g["text"].str.len().between(60, 2000))]
    parts = [g[g["rubrics"] == r].sample(per_rubric, random_state=seed) for r in RUBRICS]
    g = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return pd.DataFrame({"id": [f"geo-{i}" for i in range(len(g))], "orig": g["text"].str.replace("\\n", "\n").str.strip(),
                         "title": g["name_ru"].astype(str), "rating": g["rating"].astype(float).round().astype(int),
                         "source": g["rubrics"], "date": "", "label": g["rubrics"]})


def tickets() -> pd.DataFrame:
    t = pd.read_csv(ROOT / "data/test/tickets.csv", sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    lab = pd.read_csv(ROOT / "data/test/tickets_labels.csv", dtype=str, keep_default_na=False)
    t = t.merge(lab, on="ID")
    t = t[t["Комментарий"].str.len() > 5].reset_index(drop=True)
    rc = RuleCleaner()
    return pd.DataFrame({"id": "t-" + t["ID"], "orig": "", "dirty": t["Комментарий"], "dirty_kind": t["Отдел"],
                         "clean": [rc.clean(x).text for x in t["Комментарий"]], "title": t["Сервис"], "rating": 0,
                         "source": t["Отдел"], "date": t["Дата создания"], "label": t["topic"]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bank", type=int, default=4000)
    ap.add_argument("--per-rubric", type=int, default=350)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rc = RuleCleaner()
    for name, df in (("bank", add_variants(bank(a.n_bank, a.seed), a.seed, cleaner=rc)),
                     ("geo", add_variants(geo(a.per_rubric, a.seed), a.seed + 1, cleaner=rc)),
                     ("tickets", tickets())):
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} строк, метки: {df['label'].nunique() if df['label'].astype(bool).any() else 'нет'}, "
              f"ср. длина orig/dirty/clean: {df['orig'].str.len().mean():.0f}/{df['dirty'].str.len().mean():.0f}/{df['clean'].str.len().mean():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
