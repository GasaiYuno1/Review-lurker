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
       [--roles auto] [--topic "Тема"] [--min-topic 30] [--center-on op] [--keep "Дата,Оператор"] [--out build/run_calls]

Метки говорящих из STT бывают перепутаны (размечены каналы, а не роли): --roles auto в каждом звонке меняет роли местами,
если операторских фраз («меня зовут», «чем могу помочь», …) больше у «клиента». --topic берёт темы из готовой разметки
(темы меньше --min-topic звонков — в общую группу) вместо кластеризации звонков — и не считает эмбеддинги целых звонков.
--center-on op — эмбеддинги и центрирование только по репликам оператора (вдвое быстрее; в исследовании — по всем).
Повторы id (несколько строк на звонок) схлопываются, эмбеддинги считаются по уникальным репликам.

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
from lurker.anomaly import (AUTOANSWER, CLIENT_LABELS, OP_LABELS, OP_MARKERS, assign_topics, center, doc_max, fix_roles,  # noqa: E402
                            robust_z, sidak_score, split_dialog, turn_novelty)
from lurker.cluster import cluster_embeddings  # noqa: E402
from lurker.data import load_table  # noqa: E402
from lurker.embed import ST_MODELS, get_embedder  # noqa: E402


def stt_norm(text: str) -> str:
    """Как в эксперименте: нижний регистр, без пунктуации — вид расшифровки STT (письменные диалоги приводятся к нему)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", text.lower())).strip()


def embed(texts: list[str], model: str, cache_dir: Path, what: str, chunk: int = 50_000) -> np.ndarray:
    """Эмбеддинги уникальных текстов (реплики вроде «да», «угу», приветствия повторяются тысячи раз) с кэшем на диске.
    Кэш пишется кусками по `chunk` текстов: прерванный прогон при повторном запуске продолжается с места остановки.
    Рядом с кэшем — список его текстов (`.texts.pkl`): повторный прогон с другим набором реплик (другие роли, другой
    период) берёт готовые эмбеддинги из прежних кэшей той же модели и считает только новые тексты."""
    uniq, inv = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    uniq = uniq.tolist()
    h = hashlib.sha1("\n".join(uniq).encode("utf-8")).hexdigest()[:12]
    p = cache_dir / f"emb_{what}_{model}_{h}.npy"
    if p.exists():
        return np.load(p)[inv]
    t = time.time()
    known: dict[str, tuple[Path, int]] = {}
    reusable = model in ST_MODELS                                   # TF-IDF/гибрид обучаются на корпусе — между прогонами не переносятся
    for tp in sorted(cache_dir.glob(f"emb_*_{model}_*.texts.pkl")) if reusable else []:
        ep = tp.with_name(tp.name[:-len(".texts.pkl")] + ".npy")
        if ep.exists():
            for i, x in enumerate(pd.read_pickle(tp)):
                known.setdefault(x, (ep, i))
    old = [i for i, x in enumerate(uniq) if x in known]
    todo = [i for i, x in enumerate(uniq) if x not in known]
    enc = get_embedder(model).fit(uniq) if todo else None
    parts = []
    for i, a in enumerate(range(0, len(todo), chunk)):
        pp = cache_dir / f"emb_{what}_{model}_{h}.part{i:03d}.npy"
        if pp.exists():
            parts.append(np.load(pp))
            continue
        parts.append(enc.encode([uniq[j] for j in todo[a:a + chunk]]).astype(np.float32))
        np.save(pp, parts[-1])
        if len(todo) > chunk:
            print(f"  {what}: {min(a + chunk, len(todo))} из {len(todo)} ({time.time() - t:.0f} с)", flush=True)
    new = np.concatenate(parts) if parts else None
    dim = new.shape[1] if new is not None else None
    loaded: dict[Path, np.ndarray] = {}
    for ep in {known[uniq[i]][0] for i in old}:
        loaded[ep] = np.load(ep, mmap_mode="r")
        dim = loaded[ep].shape[1]
    E = np.zeros((len(uniq), dim or 1), dtype=np.float32)
    if new is not None:
        E[todo] = new
    for i in old:
        ep, j = known[uniq[i]]
        E[i] = loaded[ep][j]
    np.save(p, E)
    pd.to_pickle(uniq, p.with_name(p.name[:-len(".npy")] + ".texts.pkl"))
    for pp in cache_dir.glob(f"emb_{what}_{model}_{h}.part*.npy"):
        pp.unlink()
    print(f"эмбеддинги {what} {model}: {len(uniq)} уникальных из {len(texts)} (новых {len(todo)}, из прежних кэшей {len(old)}) "
          f"за {time.time() - t:.0f} с", flush=True)
    return E[inv]


def read_calls(a) -> tuple[list[str], list[list[tuple[str, str]]], pd.DataFrame]:
    """→ (id звонков, реплики [(op|cl, текст)], первая строка каждого звонка — для --topic и --keep)."""
    df = load_table(a.file)
    extra = ([a.topic] if a.topic else []) + [c.strip() for c in (a.keep or "").split(",") if c.strip()]
    for c in [a.text] + ([a.id] if a.id else []) + ([a.speaker] if a.long else []) + extra:
        if c not in df.columns:
            raise SystemExit(f"нет колонки «{c}»; есть: {', '.join(df.columns)}")
    if a.long:
        op_values = {v.strip().lower() for v in a.op_values.split(",")}
        calls: dict[str, list] = {}
        for cid, sp, tx in zip(df[a.id].astype(str), df[a.speaker].astype(str), df[a.text].fillna("").astype(str)):
            if tx.strip():
                calls.setdefault(cid, []).append(("op" if sp.strip().lower() in op_values else "cl", " ".join(tx.split())))
        first = df.assign(_id=df[a.id].astype(str)).drop_duplicates("_id").set_index("_id")
        return list(calls), list(calls.values()), first.loc[list(calls)].reset_index(drop=True)
    if a.id:
        n = len(df)
        df = df.assign(_id=df[a.id].astype(str)).drop_duplicates("_id").reset_index(drop=True)
        if len(df) < n:
            print(f"повторы id схлопнуты: {n} строк → {len(df)} звонков", flush=True)
        ids = df["_id"].tolist()
    else:
        ids = [str(i) for i in range(len(df))]
    op = [x.strip() for x in a.op_labels.split(",")] if a.op_labels else OP_LABELS
    cl = [x.strip() for x in a.client_labels.split(",")] if a.client_labels else CLIENT_LABELS
    return ids, [split_dialog(t, op, cl) for t in df[a.text].fillna("").astype(str)], df


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
    ap.add_argument("--z", type=float, default=2.0, help="порог «помечен» по робастному z внутри темы (оценка max)")
    ap.add_argument("--alpha", type=float, default=0.01, help="порог «помечен» для оценки sidak: p звонка < alpha")
    ap.add_argument("--keep-punct", action="store_true", help="не приводить реплики к виду STT (регистр, пунктуация)")
    ap.add_argument("--roles", choices=("labels", "auto"), default="labels",
                    help="auto — роли по содержанию реплик, если метки говорящих перепутаны (lurker.anomaly.fix_roles)")
    ap.add_argument("--op-markers", help="регулярное выражение операторских фраз для --roles auto (по умолчанию OP_MARKERS)")
    ap.add_argument("--topic", help="колонка с темой звонка из разметки — вместо кластеризации")
    ap.add_argument("--min-topic", type=int, default=30, help="темы меньше стольких звонков (при --topic) — в общую группу")
    ap.add_argument("--center-on", choices=("all", "op"), default="all",
                    help="центрировать по всем репликам (как в исследовании) или только по репликам оператора (вдвое быстрее)")
    ap.add_argument("--score", choices=("max", "sidak"), default="max",
                    help="оценка звонка: max — самая необычная реплика; sidak — она же с поправкой на число реплик (длинные "
                         "звонки не всплывают только из-за длины)")
    ap.add_argument("--min-words", type=int, default=1, help="реплики оператора короче стольких слов не оцениваются (обрывки STT)")
    ap.add_argument("--keep", help="колонки исходной таблицы через запятую — перенести в calls_scored.csv (дата, оператор, …)")
    ap.add_argument("--out", default=str(ROOT / "build" / "run_calls"))
    a = ap.parse_args(argv)
    if a.long and not (a.id and a.speaker):
        raise SystemExit("--long требует --id и --speaker")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ids, calls, meta = read_calls(a)
    swapped = np.zeros(len(calls), dtype=bool)
    if a.roles == "auto":
        rx = re.compile(a.op_markers, re.IGNORECASE) if a.op_markers else OP_MARKERS
        for i, t in enumerate(calls):
            calls[i], swapped[i] = fix_roles(t, rx)
        print(f"роли поменяны местами в {swapped.mean():.0%} звонков", flush=True)
    auto = [len(t) <= 10 and bool(AUTOANSWER.search(" ".join(x for _, x in t))) for t in calls]
    if any(auto):
        print(f"автоответчик или голосовой секретарь вместо разговора: {sum(auto)} звонков — пропущены", flush=True)
    keep = [i for i, t in enumerate(calls) if not auto[i] and any(s == "op" for s, _ in t)]
    if len(keep) < len(calls):
        n_noop = len(calls) - len(keep) - sum(auto)
        if n_noop:
            print(f"без реплик оператора (не распознаны метки говорящих?): {n_noop} звонков — пропущены", flush=True)
    ids, calls, swapped, meta = [ids[i] for i in keep], [calls[i] for i in keep], swapped[keep], meta.iloc[keep].reset_index(drop=True)
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

    op = speaker == "op"
    if a.topic:
        name = meta[a.topic].fillna("—").astype(str)
        big = name.map(name.value_counts()) >= a.min_topic
        name = name.where(big, "прочее (малые темы)")
        codes, uniq = pd.factorize(name)
        topics = np.asarray(codes)
        kw = dict(enumerate(uniq))
        print(f"тем из «{a.topic}»: {len(uniq)} (звонков в малых темах: {int((~big).sum())})", flush=True)
    else:
        E = embed(docs, a.model, out, "calls")
        res = cluster_embeddings(E, docs, method="umap_hdbscan", min_cluster_size=a.mcs or None, outlier_threshold=-1.0)
        topics = assign_topics(E, res.labels)
        kw = {tp.id: ", ".join([k for k in tp.keywords if k not in ("оператор", "клиент")][:6]) for tp in res.topics}
    nov = np.zeros(len(turns))
    if a.center_on == "op":
        nov[op] = turn_novelty(center(embed([x for x, o in zip(turns, op) if o], a.model, out, "opturns")), owner[op], topics, k=10)
    else:
        nov[op] = turn_novelty(center(embed(turns, a.model, out, "turns"))[op], owner[op], topics, k=10)
    elig = op & (np.array([len(x.split()) for x in turns]) >= a.min_words)
    if a.score == "sidak":
        score = sidak_score(nov[elig], owner[elig], topics, len(calls))
    else:
        score = doc_max(nov[elig], owner[elig], len(calls))
    z = robust_z(score, topics)
    # sidak — уже −log10 p внутри темы (при отсутствии аномалий p равномерно): помечать по p, а не по z — у этой оценки
    # длинный правый хвост, и z ≥ 2 метит ~15% звонков
    flag = score >= -np.log10(a.alpha) if a.score == "sidak" else z >= a.z
    rule = f"p < {a.alpha}" if a.score == "sidak" else f"z ≥ {a.z}"

    keep_cols = [c.strip() for c in (a.keep or "").split(",") if c.strip()]
    rows = []
    for d in range(len(calls)):
        m = np.where((owner == d) & elig)[0]
        if not len(m):
            m = np.where((owner == d) & op)[0]
        w = m[np.argmax(nov[m])]                                   # самая «чужая» реплика оператора
        prev = [x for s, x in calls[d][:pos[w]] if s == "cl"]
        rows.append({"id": ids[d], "topic": int(topics[d]), "topic_keywords": kw.get(int(topics[d]), ""), "z": round(float(z[d]), 2),
                     "score": round(float(score[d]), 4), "flagged": bool(flag[d]), "worst_operator_turn": shown[w],
                     "client_before": prev[-1] if prev else "", "n_turns": len(calls[d]), "roles_swapped": bool(swapped[d]),
                     **{c: meta.at[d, c] for c in keep_cols}})
    sc = pd.DataFrame(rows).sort_values("z", ascending=False)
    sc.to_csv(out / "calls_scored.csv", index=False)
    tp = sc.groupby(["topic", "topic_keywords"]).agg(calls=("id", "size"), flagged=("flagged", "mean")).reset_index()
    tp.sort_values("calls", ascending=False).to_csv(out / "topics.csv", index=False)

    md = [f"# Подозрительные звонки: {Path(a.file).name}", "",
          f"{len(calls)} звонков, тем {len(tp)}, модель {a.model}, оценка {a.score}. Помечено {rule}: {int(sc.flagged.sum())} "
          f"({sc.flagged.mean():.1%}). Ниже — топ-{min(a.top, len(sc))} по z (внутри своей темы).", ""]
    for _, r in sc.head(a.top).iterrows():
        md += [f"## {r['id']} — z = {r['z']}", f"Тема: {r['topic_keywords']}", ""]
        if r["client_before"]:
            md.append(f"> Клиент: {r['client_before'][:300]}")
        md += [f"> **Оператор: {r['worst_operator_turn'][:400]}**", ""]
    (out / "top.md").write_text("\n".join(md), encoding="utf-8")
    print(f"помечено {int(sc.flagged.sum())} из {len(sc)} ({rule}) → {out}/calls_scored.csv, topics.csv, top.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
