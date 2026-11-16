"""Командная строка (без UI):
  python3 -m lurker.cli columns <файл>                                   # угаданный маппинг колонок
  python3 -m lurker.cli analyze <файл> [--freq W] [--prompt "вопрос"] [--criteria критерии.txt] [--out отчёт.md]
                                       [--json темы.json] [--save анализ.pkl] [--map text=Комментарий,date=Дата]
                                       [--backend hybrid] [--method umap_hdbscan] [--max-topics N] [--llm]
  python3 -m lurker.cli ask <анализ.pkl> "вопрос" [--freq M] [--threshold 0.5] [--llm]
  python3 -m lurker.cli [--config configs/gemini.yaml] llmcheck [--n 20] [--data data/test/tickets.csv]   # проверка LLM-слоя
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import ColumnMap, guess_columns, load_table, normalize
from .llm import LLMClient
from .period import describe
from .pipeline import load, run_analysis, save
from .preclass import parse_criteria
from .summarize import report_markdown, to_json, topics_table


def make_llm(cfg, enabled: bool):
    if not enabled:
        return None
    if not cfg.llm.api_key and "localhost" not in cfg.llm.base_url and "127.0.0.1" not in cfg.llm.base_url:
        print(f"LLM выключен: нет ключа в переменной {cfg.llm.api_key_env}", file=sys.stderr)
        return None
    return LLMClient(cfg.llm, cache_dir=cfg.cache.dir)


def parse_map(spec: str | None, df: pd.DataFrame) -> ColumnMap:
    cm = guess_columns(df) if not spec else None
    if spec:
        kv = dict(p.split("=", 1) for p in spec.split(","))
        base = {}
        try:
            base = guess_columns(df).as_dict()
        except ValueError:
            pass
        base.update(kv)
        cm = ColumnMap(**base)
    for role in ("text", "date"):
        if getattr(cm, role) not in df.columns:
            raise SystemExit(f"колонка для {role} не найдена: {getattr(cm, role)!r}; есть: {list(df.columns)}")
    return cm


def cmd_columns(a) -> int:
    df = load_table(a.file)
    cm = guess_columns(df)
    print(f"строк: {len(df)}, колонок: {len(df.columns)}")
    for k, v in cm.as_dict().items():
        print(f"  {k:11} -> {v}")
    return 0


def cmd_analyze(a) -> int:
    cfg = load_config(a.config)
    llm = make_llm(cfg, a.llm)
    raw = load_table(a.file)
    cm = parse_map(a.map, raw)
    df = normalize(raw, cm)
    print(f"файл: {a.file}, строк {len(df)}, маппинг: " + ", ".join(f"{k}={v}" for k, v in cm.as_dict().items() if v))
    filters = dict(f.split("=", 1) for f in a.filter) if a.filter else None
    if filters:
        from .dynamics import apply_filters
        df = apply_filters(df, filters)
        print(f"после фильтра {filters}: {len(df)} строк")
    criteria = parse_criteria(Path(a.criteria).read_text(encoding="utf-8")) if a.criteria else None
    an = run_analysis(df, cfg, llm=llm, freq=a.freq, backend=a.backend, method=a.method, max_topics=a.max_topics,
                      criteria=criteria, llm_clean=a.llm_clean, log=print)
    print(f"\nдаты: {an.df['date'].min():%d.%m.%Y} – {an.df['date'].max():%d.%m.%Y}; тем: {len(an.clusters.topics)}; прочее: {an.clusters.n_outliers}")
    print("\nЧто изменилось:")
    for f in an.findings[:15]:
        print("  -", f.describe(an.topic_name(f.topic)))
    print("\nТемы:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(topics_table(an.summaries).drop(columns=["содержание"]).head(40).to_string(index=False))
    if criteria is not None:
        print("\nПредклассификация:", an.df["label"].value_counts().to_dict())
    if a.prompt:
        print_query(an, a.prompt, a.threshold, a.freq, llm, criteria)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(report_markdown(an.summaries, an.findings, an.names), encoding="utf-8")
        print(f"\nотчёт: {a.out}")
    if a.json:
        Path(a.json).write_text(to_json(an.summaries), encoding="utf-8")
    if a.save:
        save(an, a.save)
        print(f"анализ сохранён: {a.save}")
    return 0


def print_query(an, prompt: str, threshold: float, freq: str, llm, criteria=None) -> None:
    try:  # сценарий «вопрос к данным» — отделяемый подпакет lurker.service
        from .service.assistant import answer
        from .service.query import run_query
    except ImportError:
        raise SystemExit("подпакет lurker.service (сценарий «вопрос к данным») не установлен")
    q = run_query(an, prompt, threshold=threshold, freq=freq, llm=llm, criteria=criteria)
    print(f"\nВопрос: {prompt}\n{q.text}")
    fmt = {"D": "%d.%m", "W": "%d.%m", "M": "%m.%Y"}.get(freq, "%d.%m.%Y")
    print("По периодам (релевантных / всего): " + ", ".join(f"{i.strftime(fmt)}: {int(r['count'])}/{int(r['total'])}" for i, r in q.series.iterrows()))
    if q.compare is not None:
        print("\nСравнение периодов:")
        print(q.compare.to_string(float_format=lambda x: f"{x:.3f}"))
    if q.subsummaries:
        print("\nТематики внутри выборки:")
        for s in q.subsummaries:
            print(f"  {s.name:40} {s.count:5d}  {'; '.join('«' + t + '»' for _, t in s.quotes[:2])[:140]}")
    if llm is not None and q.n_relevant:
        ans = answer(prompt, an.df[q.mask].reset_index(drop=True), an.E[q.mask], an.embedder, llm=llm, freq=freq, names=an.names)
        print(f"\nОтвет LLM:\n{ans.text}")


def cmd_ask(a) -> int:
    cfg = load_config(a.config)
    an = load(a.analysis)
    print_query(an, a.question, a.threshold, a.freq, make_llm(cfg, a.llm))
    return 0


def cmd_llmcheck(a) -> int:
    """Пинг endpoint и прогон всех LLM-шагов на N тестовых комментариях: очистка, критерии из вопроса,
    названия тем, саммари, период из промпта, ответ ассистента. Печатает статус, время и токены по шагам."""
    import time
    from .clean import LLMCleaner, RuleCleaner
    from .cluster import cluster_embeddings
    from .embed import get_embedder
    from .llm import FakeLLM, parse_json
    from .period import llm_periods
    from .preclass import llm_criteria
    from .summarize import llm_summary, name_topics

    cfg = load_config(a.config)
    if a.fake:
        llm = FakeLLM(_fake_reply)
        print("режим --fake: заглушка вместо модели")
    else:
        print(f"endpoint: {cfg.llm.base_url}  модель: {cfg.llm.model}  ключ: {'есть' if cfg.llm.api_key else 'НЕТ (' + cfg.llm.api_key_env + ')'}"
              f"  лимит: {cfg.llm.rate_limit_rpm or 'нет'} req/min")
        llm = LLMClient(cfg.llm, cache_dir=None)
        t = time.time()
        ping = llm.ping()
        print(f"GET /models: {'ok' if ping['ok'] else 'нет'} (HTTP {ping['status']}, {time.time() - t:.1f} с)"
              + (f", моделей: {len(ping['models'])}" + (f", среди них {cfg.llm.model}" if cfg.llm.model in ping["models"] else "") if ping["ok"] else f": {ping['error'][:200]}"))
        t = time.time()
        try:
            reply = llm.chat([{"role": "user", "content": "Ответь одним словом: столица Франции?"}], max_tokens=20)
            print(f"chat: ok ({time.time() - t:.1f} с): {reply.strip()[:60]!r}")
        except Exception as e:
            print(f"chat: ОШИБКА — {e}")
            return 1

    raw = load_table(a.data)
    df = normalize(raw, guess_columns(raw))
    df = df[df["text"].str.len() > 20]
    smm = df[df["department"].str.lower().str.contains("smm")] if "department" in df.columns else df.iloc[0:0]
    sample = pd.concat([smm.head(a.n // 2), df.drop(smm.index).head(a.n - min(len(smm), a.n // 2))]).head(a.n)
    texts_raw = sample["text"].tolist()
    ref = df["date"].max().date()
    steps = []

    def step(name, fn):
        t = time.time()
        before = dict(llm.usage) if hasattr(llm, "usage") else {}
        try:
            out = fn()
            ok, note = True, out
        except Exception as e:
            ok, note = False, f"{type(e).__name__}: {str(e)[:160]}"
        dt = time.time() - t
        tok = ""
        if hasattr(llm, "usage"):
            tok = f" токены {llm.usage['prompt_tokens'] - before.get('prompt_tokens', 0)}/{llm.usage['completion_tokens'] - before.get('completion_tokens', 0)}"
        steps.append((name, ok))
        print(f"\n[{'ok' if ok else 'СБОЙ'}] {name} ({dt:.1f} с{tok})")
        print("   " + str(note).replace("\n", "\n   ")[:900])

    def do_clean():
        cleaner = LLMCleaner(llm, batch_size=cfg.llm.batch_size, prefilter=RuleCleaner())
        res = cleaner.clean_many(texts_raw)
        if cleaner.failed_batches:
            raise RuntimeError(f"батчей со сбоем: {cleaner.failed_batches}")
        changed = sum(1 for r, t in zip(res, texts_raw) if r.text != RuleCleaner()._tidy(t))
        lines = [f"изменено LLM относительно правил: {sum(1 for r in res if r.removed.get('llm'))}/{len(res)}"]
        for r, t in list(zip(res, texts_raw))[:3]:
            lines.append(f"- {t[:110]!r}\n  → {r.text[:110]!r}")
        return "\n".join(lines)

    clean_texts_local = [RuleCleaner().clean(t).text for t in texts_raw]

    def do_criteria():
        crit = llm_criteria("как часто нас оскорбляли и угрожали физической расправой?", llm, clean_texts_local)
        if not crit.labels:
            raise RuntimeError("метки не разобраны")
        return "\n".join(f"- {l.name}: {l.description[:80]} | примеры: {len(l.examples)} | слова: {', '.join(l.keywords[:6])}" for l in crit.labels)

    def do_names():
        emb = get_embedder("tfidf", dim=8).fit(clean_texts_local)
        E = emb.encode(clean_texts_local)
        res = cluster_embeddings(E, clean_texts_local, method="agglomerative", min_cluster_size=2, distance_threshold=0.7)
        names = name_topics(res, clean_texts_local, llm)
        if not names:
            raise RuntimeError("названий нет")
        return "\n".join(f"- {t.id}: {names.get(t.id, '—')}  (слова: {', '.join(t.keywords[:4])})" for t in res.topics[:6])

    def do_summary():
        s = llm_summary(clean_texts_local[:15], llm, "о чём жалуются?", total=len(df))
        if not s.get("темы") and not s.get("выводы"):
            raise RuntimeError("пустой ответ")
        return "\n".join([f"- {t.get('тема')}: {t.get('частота')} — {str(t.get('содержание'))[:100]}" for t in s["темы"][:4]]
                          + [f"вывод: {v}" for v in s["выводы"][:3]])

    def do_period():
        ps = llm_periods("сравни, что говорили про приложение в январе и в июле", ref, llm)
        if len(ps) != 2:
            raise RuntimeError(f"ожидал 2 периода, получил {len(ps)}: {ps}")
        return "; ".join(f"{p.label}: {p.start}–{p.end}" for p in ps)

    def do_answer():
        from .service.assistant import answer
        sub = df.head(200).reset_index(drop=True)
        emb = get_embedder("tfidf", dim=16).fit(sub["text"].tolist())
        sub["clean"] = clean_texts_local[:len(sub)] + sub["text"].tolist()[len(clean_texts_local):]
        ans = answer("что говорят про домофон?", sub, emb.encode(sub["clean"].tolist()), emb, llm=llm, k=8)
        return ans.text

    step("очистка few-shot (LLMCleaner)", do_clean)
    step("критерии из вопроса (llm_criteria)", do_criteria)
    step("названия тем (name_topics)", do_names)
    step("саммари выборки (llm_summary)", do_summary)
    step("период из промпта (llm_periods)", do_period)
    step("ответ ассистента (RAG)", do_answer)
    ok = sum(1 for _, s in steps if s)
    print(f"\nитог: {ok}/{len(steps)} шагов ок; вызовов модели: {getattr(llm, 'calls', '?')}"
          + (f"; токены всего {llm.usage['prompt_tokens']}/{llm.usage['completion_tokens']}" if hasattr(llm, "usage") else "")
          + (("; json_mode выключен в конфиге" if not cfg.llm.json_mode else
              f"; json_mode {'поддерживается' if llm._json_ok else 'ОТКЛЮЧЁН (endpoint вернул 400)'}") if hasattr(llm, "_json_ok") else ""))
    return 0 if ok == len(steps) else 1


def _fake_reply(msgs):
    """Правдоподобные ответы заглушки для --fake: по содержимому последнего сообщения."""
    import json as _json
    content = msgs[-1]["content"]
    if "periods" in content:
        return '{"periods": [{"start": "2026-01-01", "end": "2026-01-31", "label": "январь"}, {"start": "2026-07-01", "end": "2026-07-31", "label": "июль"}], "question": "приложение"}'
    if '"labels"' in content:
        return '{"labels": [{"name": "оскорбления и угрозы", "description": "абонент оскорбляет или угрожает", "examples": ["угрожал расправой"], "keywords": ["угрожать", "оскорблять"]}]}'
    if "названия" in content:
        return '{"названия": {"0": "Проблемы с интернетом", "1": "Оплата", "2": "Домофон", "3": "ТВ", "4": "Прочее", "5": "Ещё"}}'
    if '"темы"' in content:
        return '{"темы": [{"тема": "интернет", "частота": "5 из 15", "содержание": "нет связи по вечерам"}], "выводы": ["жалуются на скорость"]}'
    if content.startswith("{"):  # батч очистки
        try:
            batch = _json.loads(content)
            return _json.dumps({k: v.lower() for k, v in batch.items()}, ensure_ascii=False)
        except Exception:
            pass
    return "Ответ заглушки."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lurker", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("columns")
    p.add_argument("file")
    p.set_defaults(fn=cmd_columns)
    p = sub.add_parser("analyze")
    p.add_argument("file")
    p.add_argument("--freq", default="W", choices=["D", "W", "M"])
    p.add_argument("--prompt")
    p.add_argument("--criteria", help="файл с критериями предклассификации (см. preclass.parse_criteria)")
    p.add_argument("--out")
    p.add_argument("--json")
    p.add_argument("--save")
    p.add_argument("--map", help="text=...,date=...,service=...,author=...")
    p.add_argument("--filter", action="append", help="колонка=значение (можно несколько)")
    p.add_argument("--backend")
    p.add_argument("--method")
    p.add_argument("--max-topics", type=int)
    p.add_argument("--llm", action="store_true", help="использовать LLM из конфига (нужен ключ)")
    p.add_argument("--llm-clean", action="store_true", help="очистка через LLM (поверх правил)")
    p.add_argument("--threshold", type=float, default=0.5, help="порог отбора по смыслу для --prompt")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("ask")
    p.add_argument("analysis")
    p.add_argument("question")
    p.add_argument("--freq", default="M", choices=["D", "W", "M"])
    p.add_argument("--llm", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("llmcheck", help="проверка endpoint и всех LLM-шагов на тестовых комментариях")
    p.add_argument("--data", default="data/test/tickets.csv")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--fake", action="store_true", help="заглушка вместо модели (проверка самой команды)")
    p.set_defaults(fn=cmd_llmcheck)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
