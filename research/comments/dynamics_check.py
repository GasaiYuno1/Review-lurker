#!/usr/bin/env python3
"""Задача 1, динамика (вопрос, с которого началось ТЗ: «стали ли чаще жаловаться на домофон в приложении?», и «какие
темы появляются или становятся заметнее»). В синтетические тикеты при генерации заложены события по датам
(data/test/tickets_labels.csv, колонка trend): всплеск жалоб на домофон в приложении в марте, рост проблем с оплатой
с июня, плавный рост расторжений, зимний сезон низкой скорости, рост угроз, новая тема в августе.
Проверяем: если взять кластеры (очищенный текст, USER-bge-m3, без всякой разметки), посчитать их по неделям и прогнать
поиск изменений (lurker.dynamics.detect_changes: всплеск, новая тема, рост), находятся ли заложенные события, и насколько
недельный ряд кластера повторяет настоящий ряд темы.
python3 research/comments/dynamics_check.py → research/results/dynamics_check.csv, dynamics_findings.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BUILD, DATA, RESULTS, ROOT  # noqa: E402
from lurker.dynamics import detect_changes, timeline  # noqa: E402

KEY = "tickets__clean__user-bge-m3"
EXPECT = {"всплеск_март": "spike", "новая_тема_август": "new", "рост_с_июня": "growth", "рост_плавный": "growth",
          "рост_негатива_2мес": "spike|growth", "сезон_зима": "decline"}   # зимний сезон в ряду янв–сен виден как спад


def main() -> int:
    t = pd.read_parquet(DATA / "comments" / "tickets.parquet")
    lab = pd.read_csv(ROOT / "data" / "test" / "tickets_labels.csv")[["ID", "trend"]]
    lab["id"] = "t-" + lab["ID"].astype(str)
    t = t.merge(lab[["id", "trend"]], on="id", how="left")
    t["date"] = pd.to_datetime(t["date"])
    t["cluster"] = np.load(BUILD / "clusters" / f"{KEY}.npy")
    names = {c["id"]: c["name"] for c in json.loads((RESULTS / "clusters_tickets.json").read_text(encoding="utf-8"))["clusters"]}

    tl_c = timeline(t[t.cluster >= 0], topic_col="cluster", freq="W")
    tl_t = timeline(t, topic_col="label", freq="W")
    found_c = detect_changes(tl_c, max_findings=100)
    found_t = detect_changes(tl_t, max_findings=100)
    pd.DataFrame([{"source": src, "kind": f.kind, "topic": f.topic, "name": names.get(f.topic, "") if src == "кластеры" else "",
                   "period": f.period, "count": f.count, "baseline": f.baseline, "factor": round(f.factor, 2) if np.isfinite(f.factor) else None}
                  for src, fs in (("кластеры", found_c), ("настоящие темы", found_t)) for f in fs]).to_csv(
        RESULTS / "dynamics_findings.csv", index=False)

    dom = t[t.cluster >= 0].groupby("cluster")["label"].agg(lambda x: x.value_counts().index[0])     # главная тема кластера
    trend_of = t.dropna(subset=["trend"]).query("trend != ''").groupby("label")["trend"].first()
    rows = []
    for topic, trend in trend_of.items():
        kinds = EXPECT.get(trend, "").split("|")
        cls = [int(c) for c in dom[dom == topic].index]                # все кластеры, где эта тема главная
        hits_c = [f for f in found_c if f.topic in cls and f.kind in kinds]
        hits_t = [f for f in found_t if f.topic == topic and f.kind in kinds]
        truth = tl_t[topic].reindex(tl_c.index, fill_value=0) if topic in tl_t else None
        series = tl_c[[c for c in cls if c in tl_c]].sum(axis=1) if cls else None
        corr = float(np.corrcoef(truth, series)[0, 1]) if truth is not None and series is not None and series.std() > 0 else np.nan
        rows.append({"заложено": trend, "тема": topic, "тикетов темы": int((t.label == topic).sum()),
                     "кластеров с этой темой главной": len(cls),
                     "доля темы в этих кластерах": round(float(t[t.label == topic].cluster.isin(cls).mean()), 2),
                     "ожидаемая находка": EXPECT.get(trend, ""), "найдено по кластерам": bool(hits_c),
                     "найдено по настоящим темам": bool(hits_t),
                     "находка": hits_c[0].describe(names.get(hits_c[0].topic, f"кластер {hits_c[0].topic}")) if hits_c else "",
                     "корреляция недельных рядов": round(corr, 2)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "dynamics_check.csv", index=False)
    print(out.drop(columns=["находка"]).to_string(index=False))
    planted = set(trend_of.index)
    ok = [f for f in found_c if dom.get(f.topic) in planted]
    print(f"\nнаходок по кластерам: {len(found_c)}, из них на кластерах с заложенным событием: {len(ok)}; "
          f"по настоящим темам: {len(found_t)}")
    for f in found_c:
        mark = "✓" if dom.get(f.topic) in planted else "·"
        print("  ", mark, f.describe(names.get(f.topic, f"кластер {f.topic} ({dom.get(f.topic)})")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
