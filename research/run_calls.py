#!/usr/bin/env python3
"""Задача 2 на своих данных: расшифровки звонков → темы (эмбеддинг звонка, UMAP + HDBSCAN) → новизна каждой реплики
оператора среди реплик операторов в других звонках той же темы (центрированные эмбеддинги) → робастный z внутри темы →
топ-N подозрительных звонков с самой «чужой» репликой для контролёра качества.
Конфигурация — лучшая по итогам исследования (USER-base, turns_c; research/REPORT.md).

Расшифровка одной строкой с метками говорящих:
  python3 research/run_calls.py calls.csv --text "Диалог" [--id "ID звонка"]
Или по реплике на строку:
  python3 research/run_calls.py turns.csv --long --id call_id --speaker role --text phrase [--op-values "оператор,operator"]
Общие: [--model user-base] [--top 50] [--z 2.0] [--op-labels "Оператор,Менеджер"] [--client-labels "Клиент,Абонент"]
       [--out build/run_calls]

Выход (в --out):
  calls_scored.csv — все звонки: тема, z, оценка, помечен ли (z ≥ порога), самая чужая реплика оператора и реплика клиента
                     перед ней;
  topics.csv       — темы: размер, ключевые слова, доля помеченных;
  top.md           — топ-N для ручной проверки.
Порог z ≥ 2 на тестовых данных помечает ~3.5% звонков; это список на проверку человеком, а не автоматическое решение.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lurker.anomaly import CLIENT_LABELS, OP_LABELS, assign_topics, center, doc_max, robust_z, split_dialog, turn_novelty  # noqa: E402
from lurker.cluster import cluster_embeddings  # noqa: E402
from lurker.data import load_table  # noqa: E402
from lurker.embed import get_embedder  # noqa: E402


def stt_norm(text: str) -> str:
    """Как в эксперименте: нижний регистр, без пунктуации — вид расшифровки STT (письменные диалоги приводятся к нему)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", text.lower())).strip()


def embed(texts: list[str], model: str, cache_dir: Path, what: str) -> np.ndarray:
    h = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    p = cache_dir / f"emb_{what}_{model}_{h}.npy"
    if p.exists():
        return np.load(p)
    t = time.time()
    E = get_embedder(model).fit(texts).encode(texts).astype(np.float32)
    np.save(p, E)
    print(f"эмбеддинги {what} {model}: {E.shape[0]} за {time.time() - t:.0f} с", flush=True)
    return E


def read_calls(a) -> tuple[list[str], list[list[tuple[str, str]]]]:
    """→ (id звонков, реплики [(op|cl, текст)])."""
    df = load_table(a.file)
    for c in [a.text] + ([a.id] if a.id else []) + ([a.speaker] if a.long else []):
        if c not in df.columns:
            raise SystemExit(f"нет колонки «{c}»; есть: {', '.join(df.columns)}")
    if a.long:
        op_values = {v.strip().lower() for v in a.op_values.split(",")}
        calls: dict[str, list] = {}
        for cid, sp, tx in zip(df[a.id].astype(str), df[a.speaker].astype(str), df[a.text].fillna("").astype(str)):
            if tx.strip():
                calls.setdefault(cid, []).append(("op" if sp.strip().lower() in op_values else "cl", " ".join(tx.split())))
        return list(calls), list(calls.values())
    ids = df[a.id].astype(str).tolist() if a.id else [str(i) for i in range(len(df))]
    op = [x.strip() for x in a.op_labels.split(",")] if a.op_labels else OP_LABELS
    cl = [x.strip() for x in a.client_labels.split(",")] if a.client_labels else CLIENT_LABELS
    return ids, [split_dialog(t, op, cl) for t in df[a.text].fillna("").astype(str)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="CSV или XLSX")
    ap.add_argument("--text", required=True, help="колонка с расшифровкой (или с репликой при --long)")
    ap.add_argument("--id", help="колонка с id звонка (при --long обязательна)")
    ap.add_argument("--long", action="store_true", help="по реплике на строку")
    ap.add_argument("--speaker", help="колонка с говорящим (при --long)")
    ap.add_argument("--op-values", default="оператор,operator,op,менеджер,сотрудник,agent", help="значения --speaker для оператора")
    ap.add_argument("--op-labels", help="метки оператора в тексте через запятую (по умолчанию Оператор, Менеджер, …)")
    ap.add_argument("--client-labels", help="метки клиента в тексте через запятую (по умолчанию Клиент, Абонент, …)")
    ap.add_argument("--model", default="user-base")
    ap.add_argument("--mcs", type=int, default=0, help="min_cluster_size тем (0 — авто: √n/2)")
    ap.add_argument("--top", type=int, default=50, help="сколько звонков в top.md")
    ap.add_argument("--z", type=float, default=2.0, help="порог «помечен» по робастному z внутри темы")
    ap.add_argument("--keep-punct", action="store_true", help="не приводить реплики к виду STT (регистр, пунктуация)")
    ap.add_argument("--out", default=str(ROOT / "build" / "run_calls"))
    a = ap.parse_args(argv)
    if a.long and not (a.id and a.speaker):
        raise SystemExit("--long требует --id и --speaker")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ids, calls = read_calls(a)
    keep = [i for i, t in enumerate(calls) if any(s == "op" for s, _ in t)]
    if len(keep) < len(calls):
        print(f"без реплик оператора (не распознаны метки говорящих?): {len(calls) - len(keep)} звонков — пропущены", flush=True)
    ids, calls = [ids[i] for i in keep], [calls[i] for i in keep]
    if len(calls) < 20:
        raise SystemExit(f"слишком мало звонков с репликами оператора: {len(calls)}")
    norm = (lambda x: x) if a.keep_punct else stt_norm
    docs = ["\n".join(("оператор: " if s == "op" else "клиент: ") + norm(x) for s, x in t) for t in calls]
    turns = [norm(x) or "-" for t in calls for _, x in t]
    shown = [x for t in calls for _, x in t]                        # исходный текст реплик — для отчёта
    owner = np.array([d for d, t in enumerate(calls) for _ in t])
    speaker = np.array([s for t in calls for s, _ in t])
    pos = np.array([j for t in calls for j in range(len(t))])
    print(f"{len(calls)} звонков, {len(turns)} реплик, из них оператора {(speaker == 'op').sum()}", flush=True)

    E = embed(docs, a.model, out, "calls")
    T = embed(turns, a.model, out, "turns")
    res = cluster_embeddings(E, docs, method="umap_hdbscan", min_cluster_size=a.mcs or None, outlier_threshold=-1.0)
    topics = assign_topics(E, res.labels)
    op = speaker == "op"
    nov = np.zeros(len(turns))
    nov[op] = turn_novelty(center(T)[op], owner[op], topics, k=10)
    score = doc_max(nov[op], owner[op], len(calls))
    z = robust_z(score, topics)

    kw = {tp.id: ", ".join([k for k in tp.keywords if k not in ("оператор", "клиент")][:6]) for tp in res.topics}
    rows = []
    for d in range(len(calls)):
        m = np.where((owner == d) & op)[0]
        w = m[np.argmax(nov[m])]                                   # самая «чужая» реплика оператора
        prev = [x for s, x in calls[d][:pos[w]] if s == "cl"]
        rows.append({"id": ids[d], "topic": int(topics[d]), "topic_keywords": kw.get(int(topics[d]), ""), "z": round(float(z[d]), 2),
                     "score": round(float(score[d]), 4), "flagged": bool(z[d] >= a.z), "worst_operator_turn": shown[w],
                     "client_before": prev[-1] if prev else "", "n_turns": len(calls[d])})
    sc = pd.DataFrame(rows).sort_values("z", ascending=False)
    sc.to_csv(out / "calls_scored.csv", index=False)
    tp = sc.groupby(["topic", "topic_keywords"]).agg(calls=("id", "size"), flagged=("flagged", "mean")).reset_index()
    tp.sort_values("calls", ascending=False).to_csv(out / "topics.csv", index=False)

    md = [f"# Подозрительные звонки: {Path(a.file).name}", "",
          f"{len(calls)} звонков, тем {len(tp)}, модель {a.model}. Помечено z ≥ {a.z}: {int(sc.flagged.sum())} "
          f"({sc.flagged.mean():.1%}). Ниже — топ-{min(a.top, len(sc))} по z (внутри своей темы).", ""]
    for _, r in sc.head(a.top).iterrows():
        md += [f"## {r['id']} — z = {r['z']}", f"Тема: {r['topic_keywords']}", ""]
        if r["client_before"]:
            md.append(f"> Клиент: {r['client_before'][:300]}")
        md += [f"> **Оператор: {r['worst_operator_turn'][:400]}**", ""]
    (out / "top.md").write_text("\n".join(md), encoding="utf-8")
    print(f"помечено {int(sc.flagged.sum())} из {len(sc)} (z ≥ {a.z}) → {out}/calls_scored.csv, topics.csv, top.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
