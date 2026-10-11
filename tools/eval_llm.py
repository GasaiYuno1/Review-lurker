#!/usr/bin/env python3
"""Что даёт LLM сверх правил — замер на тестовых данных со скрытой разметкой (нужен работающий endpoint из конфига).
  1) очистка: LLM few-shot поверх правил против одних правил — остатки ПД, длина, сохранение темы (по ключевым словам темы);
  2) критерии из вопроса: llm_criteria → run_query против понятия без LLM — точность/полнота по разметке;
  3) период из промпта: согласие LLM с разборщиком правил на 20 формулировках;
  4) названия тем и саммари — печать для глаз.
python3 tools/eval_llm.py --config configs/llamacpp.yaml [--n 40] [--prompts 4] [--skip clean,criteria,period,names]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from eval import QUERY_TRUTH, load  # noqa: E402
from lurker.clean import EMAIL_RE, PHONE_RE, URL_RE, LLMCleaner, RuleCleaner  # noqa: E402
from lurker.service.concept import build_concept, score  # noqa: E402
from lurker.config import load_config  # noqa: E402
from lurker.llm import LLMClient  # noqa: E402
from lurker.period import llm_periods, parse_periods  # noqa: E402
from lurker.pipeline import run_analysis  # noqa: E402
from lurker.preclass import llm_criteria  # noqa: E402
from lurker.service.query import run_query  # noqa: E402
from lurker.summarize import llm_summary, name_topics, sample_texts  # noqa: E402

PATR = re.compile(r"[А-ЯЁ][а-яё]+(?:ович|евич|овна|евна)\b")
PERIOD_PHRASES = ["сравни, что говорили про приложение в январе и в июле", "покажи тренд негатива за последние 3 месяца",
                  "за последние полгода", "что было с марта по май?", "жалобы за прошлый месяц", "с 1 марта по 15 апреля 2026",
                  "01.02.2026 - 28.02.2026", "в 1 квартале 2026 и во втором квартале", "за 2025 год", "в мае",
                  "за последнюю неделю", "за последние пару недель", "март-апрель 2026", "в этом месяце",
                  "что говорят про скорость", "за октябрь", "с ноября по февраль", "за последние 10 дней",
                  "как изменилась динамика домофона за полгода", "нас стали чаще оскорблять в последние 2 месяца?"]


def topic_words(df: pd.DataFrame) -> dict[str, set]:
    """Характерные леммы темы: слова, частые в теме и редкие вне неё (по очищенным правилами текстам)."""
    from lurker import text as T
    from collections import Counter
    by_topic: dict[str, Counter] = {}
    total: Counter = Counter()
    for t, txt in zip(df["topic"], df["clean"]):
        ls = set(T.lemmas(txt))
        by_topic.setdefault(t, Counter()).update(ls)
        total.update(ls)
    n_topics = len(by_topic)
    out = {}
    for t, c in by_topic.items():
        size = int((df["topic"] == t).sum())
        out[t] = {w for w, k in c.items() if k / max(size, 1) >= 0.3 and k / max(total[w], 1) >= 0.5}
    return out


def eval_clean(df: pd.DataFrame, llm, n: int) -> None:
    print(f"\n== очистка: LLM few-shot поверх правил против правил ({n} текстов, половина SMM)")
    smm = df[df["department"] == "SMM"]
    sample = pd.concat([smm.head(n // 2), df[df["department"] != "SMM"].head(n - n // 2)])
    rules = RuleCleaner()
    t = time.time()
    r_out = [rules.clean(x).text for x in sample["text"]]
    t_rules = time.time() - t
    cleaner = LLMCleaner(llm, batch_size=10, prefilter=rules)
    t = time.time()
    l_out = [r.text for r in cleaner.clean_many(sample["text"].tolist())]
    t_llm = time.time() - t
    tw = topic_words(df)

    def stats(outs):
        s = pd.Series(outs)
        keep = np.mean([bool(tw.get(tp) & set(__import__("lurker.text", fromlist=["lemmas"]).lemmas(o))) if tw.get(tp) else True
                        for tp, o in zip(sample["topic"], outs)])
        return (f"остатки: url {s.str.contains(URL_RE).mean():.2f} тел {s.str.contains(PHONE_RE).mean():.2f} email {s.str.contains(EMAIL_RE).mean():.2f} "
                f"отчество {s.str.contains(PATR).mean():.2f}; пустых {(s == '').mean():.2f}; ср. длина {s.str.len().mean():.0f}; "
                f"сохранена лексика темы {keep:.2f}")
    print(f"  правила     ({t_rules:.1f} с): " + stats(r_out))
    print(f"  правила+LLM ({t_llm:.1f} с, сбойных батчей {cleaner.failed_batches}): " + stats(l_out))
    changed = [(a, b, c) for a, b, c in zip(sample["text"], r_out, l_out) if b != c]
    print(f"  LLM изменил {len(changed)}/{n}; примеры:")
    for raw, a, b in changed[:5]:
        print(f"   raw: {raw[:100]!r}\n   правила: {a[:100]!r}\n   +LLM:    {b[:100]!r}")


def eval_criteria(df: pd.DataFrame, llm, k: int) -> None:
    print(f"\n== критерии из вопроса: понятие без LLM против критериев LLM ({k} промптов, порог 0.5)")
    an = run_analysis(df.drop(columns=[c for c in ("clean", "topic", "topic_name") if c in df.columns]).rename(columns={"topic": "_t"}),
                      cache_dir=ROOT / ".cache")
    d = df.reset_index(drop=True)
    texts = an.df["clean"].tolist()
    nonempty = (an.df["clean"].str.len() > 3).to_numpy()
    for prompt, fn in list(QUERY_TRUTH.items())[:k]:
        truth = fn(d).to_numpy() & nonempty
        c, _ = build_concept(prompt, an.embedder, texts, an.E, an.df["date"].max().date())
        pred0 = score(c, texts, an.E) >= 0.5
        t = time.time()
        def pr(pred):
            tp = int((pred & truth).sum())
            return tp / max(pred.sum(), 1), tp / max(truth.sum(), 1)
        p0, r0 = pr(pred0)
        line = f"  {prompt[:48]:48} без LLM p={p0:.2f} r={r0:.2f}"
        try:
            crit = llm_criteria(prompt, llm, an.df.loc[nonempty, "clean"].head(12).tolist())
            q = run_query(an, prompt, threshold=0.5, freq="M", llm=llm, llm_criteria_augment=True)   # промпт + дополнение критериями LLM
            pa, ra = pr(q.mask | ((q.labels != "прочее").to_numpy() & nonempty))
            q2 = run_query(an, prompt, threshold=0.5, freq="M", criteria=crit)   # только критерии LLM, все метки
            lab = q2.labels.to_numpy()
            best = max(((pr((lab == l.name) & nonempty), l.name) for l in crit.labels), key=lambda x: 2 * x[0][0] * x[0][1] / max(x[0][0] + x[0][1], 1e-9))
            (pb, rb), name = best
            line += (f" | промпт+критерии LLM p={pa:.2f} r={ra:.2f} ({time.time() - t:.0f} с) | только критерии, лучшая метка «{name}» p={pb:.2f} r={rb:.2f}"
                     f"; метки: {[l.name for l in crit.labels][:3]}")
        except Exception as e:
            line += f" | с LLM: сбой {e}"
        print(line)


def eval_period(llm) -> None:
    print("\n== период из промпта: LLM против правил")
    ref = date(2026, 9, 7)
    agree = 0
    for ph in PERIOD_PHRASES:
        a = [(p.start, p.end) for p in parse_periods(ph, ref)]
        try:
            b = [(p.start, p.end) for p in llm_periods(ph, ref, llm, date(2026, 1, 1), ref)]
        except Exception as e:
            b = f"сбой: {e}"
        ok = a == b
        agree += ok
        if not ok:
            print(f"  ≠ {ph!r}: правила {a} | LLM {b}")
    print(f"  совпало {agree}/{len(PERIOD_PHRASES)}")


def eval_names(df: pd.DataFrame, llm) -> None:
    print("\n== названия тем и саммари (для глаз)")
    an = run_analysis(df.drop(columns=[c for c in ("clean", "topic", "topic_name") if c in df.columns]).rename(columns={"topic": "_t"}),
                      cache_dir=ROOT / ".cache")
    t = time.time()
    try:
        names = name_topics(an.clusters, an.df["clean"].tolist(), llm, batch=8)
        print(f"  названий: {len(names)}/{len(an.clusters.topics)} ({time.time() - t:.0f} с)")
        for tp in an.clusters.topics[:12]:
            print(f"   {tp.id:3d} {tp.size:4d}  {names.get(tp.id, '—'):40}  ← {', '.join(tp.keywords[:5])}")
    except Exception as e:
        print(f"  названия: сбой {e}")
    t = time.time()
    try:
        s = llm_summary(sample_texts(an.df, 15), llm, "на что жалуются абоненты?", total=len(an.df))
        print(f"  саммари ({time.time() - t:.0f} с):")
        for row in s.get("темы", [])[:5]:
            print(f"   - {row.get('тема')}: {row.get('частота')} — {str(row.get('содержание'))[:120]}")
        for v in s.get("выводы", [])[:4]:
            print(f"   вывод: {v}")
    except Exception as e:
        print(f"  саммари: сбой {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=str(ROOT / "data/test/tickets.csv"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    cfg = load_config(a.config)
    llm = LLMClient(cfg.llm, cache_dir=cfg.cache.dir)
    print(f"endpoint {cfg.llm.base_url} модель {cfg.llm.model}")
    df = load(a.data)
    df["clean"] = [RuleCleaner().clean(x).text for x in df["text"]]
    skip = set(a.skip.split(",")) if a.skip else set()
    t0 = time.time()
    if "clean" not in skip:
        eval_clean(df, llm, a.n)
    if "period" not in skip:
        eval_period(llm)
    if "criteria" not in skip:
        eval_criteria(df, llm, a.prompts)
    if "names" not in skip:
        eval_names(df, llm)
    print(f"\nвсего: {time.time() - t0:.0f} с, вызовов {llm.calls}, токены {llm.usage['prompt_tokens']}/{llm.usage['completion_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
