#!/usr/bin/env python3
"""Задача 2, проверка вне выборки (аудит): выбранная конфигурация (research/run_calls.py как есть, USER-base,
новизна центрированных реплик оператора внутри темы, z ≥ 2) на ДРУГОМ корпусе — Russian MultiWOZ (ТЗ: «письменные
диалоги, но они тоже сойдут»; github.com/vonCount/Russian-MultiWOZ, MIT), 3000 однодоменных диалогов
(гостиницы, рестораны, поезда, такси, достопримечательности, больница, полиция) — объём, как у заказчика в день.

Порча и аномалии — теми же функциями и в тех же долях, что в основном эксперименте (research/calls/prepare_calls.py):
  испорченные 8% (шум STT), аномальные 6%: rude и absurd — одна реплика оператора из тех же пулов;
  splice — кусок банковского диалога (из датасета задачи 2) посреди диалога о бронировании (буквально пример из ТЗ);
  splice_near — кусок диалога другой области MultiWOZ (гостиница ↔ ресторан и т.п.).
Ничего не подбиралось на этом корпусе: метод, модель, порог — из основного эксперимента.

python3 research/calls/multiwoz_check.py prepare <каталог клона Russian-MultiWOZ>
python3 research/calls/multiwoz_check.py run          # research/run_calls.py → build/mwoz_calls/
python3 research/calls/multiwoz_check.py eval         # → research/results/mwoz_metrics.csv, mwoz_lists.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUILD, DATA, RESULTS, ROOT  # noqa: E402
from prepare_calls import ABSURD, RUDE, anomaly_line, anomaly_splice, render, stt_noise  # noqa: E402

OUT = DATA / "calls_mwoz"
N, CORRUPT, ANOMALY, SEED = 3000, 0.08, 0.06, 21
BANK_RE = r"банк|карт|кредит|вклад|счёт|счет|перевод|ипотек"


def load_mwoz(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("*/*_dialogues_*_ru.json")):
        for d in json.loads(f.read_text(encoding="utf-8")):
            services = d.get("services") or []
            turns = [["op" if t.get("speaker") == "SYSTEM" else "cl", " ".join(str(t.get("utterance", "")).split())]
                     for t in d.get("turns", [])]
            turns = [t for t in turns if t[1]]
            if len(services) == 1 and len(turns) >= 4:
                rows.append({"src_id": d.get("dialogue_id"), "domain": services[0], "turns": turns})
    return pd.DataFrame(rows).drop_duplicates("src_id")


def prepare(root: Path) -> None:
    rng = random.Random(SEED)
    full = load_mwoz(root)
    df = full.sample(n=min(N, len(full)), random_state=SEED).reset_index(drop=True)
    calls = pd.read_parquet(DATA / "calls" / "calls.parquet")
    bank = calls[(calls.kind == "normal") & calls.theme.str.lower().str.contains(BANK_RE)]
    bank_turns = [json.loads(t) for t in bank["turns"]]
    n = len(df)
    ids = list(range(n))
    rng.shuffle(ids)
    n_an, n_co = int(round(n * ANOMALY)), int(round(n * CORRUPT))
    n_far, n_near = int(round(n_an * 0.25)), int(round(n_an * 0.15))
    n_rude = n_abs = (n_an - n_far - n_near) // 2
    far, near = ids[:n_far], ids[n_far:n_far + n_near]
    rude = ids[n_far + n_near:n_far + n_near + n_rude]
    absurd = ids[n_far + n_near + n_rude:n_far + n_near + n_rude + n_abs]
    corrupt = ids[n_far + n_near + n_rude + n_abs:n_far + n_near + n_rude + n_abs + n_co]
    kind, atype = ["normal"] * n, [""] * n
    turns = [list(map(list, t)) for t in df["turns"]]
    rude_pool, absurd_pool = RUDE[:], ABSURD[:]
    rng.shuffle(rude_pool)
    rng.shuffle(absurd_pool)
    clean_ids = set(ids[n_far + n_near + n_rude + n_abs + n_co:])
    for i in far:
        turns[i] = anomaly_splice(turns[i], rng.choice(bank_turns), rng)
        kind[i], atype[i] = "anomaly", "splice"
    for i in near:
        donors = [j for j in clean_ids if df.loc[j, "domain"] != df.loc[i, "domain"]]
        turns[i] = anomaly_splice(turns[i], df.loc[rng.choice(donors), "turns"], rng)
        kind[i], atype[i] = "anomaly", "splice_near"
    for i in rude:
        turns[i] = anomaly_line(turns[i], rude_pool, rng, 1)
        kind[i], atype[i] = "anomaly", "rude"
    for i in absurd:
        turns[i] = anomaly_line(turns[i], absurd_pool, rng, 1)
        kind[i], atype[i] = "anomaly", "absurd"
    for i in corrupt:
        turns[i] = stt_noise(turns[i], rng)
        kind[i] = "corrupted"
    df["id"] = [f"mwoz-{i:05d}" for i in range(n)]
    df["kind"], df["anomaly_type"] = kind, atype
    df["text"] = [render(t) for t in turns]
    OUT.mkdir(parents=True, exist_ok=True)
    df[["id", "src_id", "domain", "kind", "anomaly_type", "text"]].to_csv(OUT / "calls.csv", index=False)
    lists = {"corrupted": df.loc[df.kind == "corrupted", "id"].tolist(),
             "anomalous": {t: df.loc[df.anomaly_type == t, "id"].tolist() for t in ("splice", "splice_near", "rude", "absurd")}}
    (RESULTS / "mwoz_lists.json").write_text(json.dumps(lists, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"MultiWOZ: однодоменных {len(full)}, взято {n}; области {df.domain.value_counts().to_dict()}")
    print(f"испорчено {len(corrupt)}, аномалий {n_an}: splice (банк) {n_far}, splice_near {n_near}, rude {n_rude}, absurd {n_abs}")


def run() -> None:
    sys.path.insert(0, str(ROOT / "research"))
    import run_calls
    run_calls.main([str(OUT / "calls.csv"), "--text", "text", "--id", "id", "--out", str(BUILD / "mwoz_calls")])


def evaluate() -> None:
    from lurker.anomaly import ranking_metrics
    sc = pd.read_csv(BUILD / "mwoz_calls" / "calls_scored.csv")
    meta = pd.read_csv(OUT / "calls.csv")[["id", "domain", "kind", "anomaly_type"]]
    m = sc.merge(meta, on="id")
    pos, cor = (m.kind == "anomaly").to_numpy(), (m.kind == "corrupted").to_numpy()
    rows = []
    base = ranking_metrics(m.z.to_numpy(), pos, ks=(40, 80, 120, 200), corrupted=cor)
    rows.append({"subset": "все аномалии", "n_pos": int(pos.sum()), "base_rate": round(pos.mean(), 4), **{k: round(v, 4) for k, v in base.items()}})
    for t in ("rude", "absurd", "splice", "splice_near"):
        keep = (m.anomaly_type == t) | (m.kind != "anomaly")
        p = (m.anomaly_type[keep] == t).to_numpy()
        r = ranking_metrics(m.z[keep].to_numpy(), p, ks=(40,), corrupted=(m.kind[keep] == "corrupted").to_numpy())
        rows.append({"subset": t, "n_pos": int(p.sum()), "base_rate": round(p.mean(), 4), **{k: round(v, 4) for k, v in r.items()}})
    flagged = m.flagged.to_numpy()
    rows.append({"subset": "порог z ≥ 2", "n_pos": int(pos.sum()), "flagged_share": round(flagged.mean(), 4),
                 "precision": round(pos[flagged].mean(), 4), "recall": round(flagged[pos].mean(), 4),
                 "noise_flagged": round(cor[flagged].mean(), 4)})
    from sklearn.metrics import normalized_mutual_info_score
    rows.append({"subset": "темы против областей", "nmi": round(normalized_mutual_info_score(m.domain, m.topic), 4),
                 "n_topics": int(m.topic.nunique())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "mwoz_metrics.csv", index=False)
    print(out.to_string(index=False))


def main() -> int:
    step = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if step == "prepare":
        prepare(Path(sys.argv[2]))
    elif step == "run":
        run()
    else:
        evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
