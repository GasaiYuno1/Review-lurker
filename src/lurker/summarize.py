"""Саммари тем и выборок (ТЗ п. 2, 5): экстрактивно без LLM и через LLM по небольшой выборке.

Экстрактивно: название темы — ключевые слова, содержание — представительные цитаты, частота и доля, разбивка по сервисам/отделам,
динамика (находки detect_changes). Через LLM: названия тем по ключевым словам и представителям; таблица
«тема / частота упоминаний / содержание» и выводы по выборке ≤ max_docs текстов, стратифицированной по времени;
сравнение двух периодов «что изменилось».
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .cluster import ClusterResult
from .dynamics import Finding


@dataclass
class TopicSummary:
    id: int
    name: str
    count: int
    share: float
    keywords: list[str]
    quotes: list[tuple[str, str]]              # (дата, текст)
    by: dict[str, dict[str, int]] = field(default_factory=dict)   # колонка → {значение: число}
    findings: list[Finding] = field(default_factory=list)
    content: str = ""                           # содержание (LLM) или пусто

    def trend_text(self) -> str:
        return "; ".join(f.describe(self.name) for f in self.findings) or "без заметных изменений"


def sample_texts(df: pd.DataFrame, n: int = 60, col: str = "clean") -> list[str]:
    """Равномерно по времени, без дублей, непустые."""
    d = df[df[col].str.len() > 3].sort_values("date")
    d = d.drop_duplicates(subset=[col])
    if len(d) <= n:
        return d[col].tolist()
    idx = np.linspace(0, len(d) - 1, n).round().astype(int)
    return d.iloc[idx][col].tolist()


def summarize_topics(df: pd.DataFrame, result: ClusterResult, findings: Sequence[Finding] = (), topic_col: str = "topic",
                     by_cols: Sequence[str] = ("service", "department"), n_quotes: int = 3, text_col: str = "clean",
                     raw_col: str = "text") -> list[TopicSummary]:
    total = max(len(df), 1)
    out = []
    for tp in result.topics:
        sub = df[df[topic_col] == tp.id]
        quotes = []
        for i in tp.representatives[:n_quotes]:
            if i < len(df):
                row = df.iloc[i]
                d = row["date"].strftime("%d.%m.%Y") if pd.notna(row["date"]) else ""
                quotes.append((d, str(row[text_col])))
        by = {c: sub[c].value_counts().head(5).to_dict() for c in by_cols if c in sub.columns and sub[c].astype(str).str.len().gt(0).any()}
        out.append(TopicSummary(tp.id, tp.label, tp.size, tp.size / total, tp.keywords, quotes, by,
                                [f for f in findings if f.topic == tp.id]))
    other = int((df[topic_col] == -1).sum())
    if other:
        sub = df[df[topic_col] == -1]
        qs = [(r["date"].strftime("%d.%m.%Y") if pd.notna(r["date"]) else "", str(r[text_col])) for _, r in sub.head(n_quotes).iterrows()]
        out.append(TopicSummary(-1, "прочее", other, other / total, [], qs, {}, [f for f in findings if f.topic == -1]))
    return out


# ---------- LLM ----------

NAME_PROMPT = """Дай каждой теме короткое название (2–5 слов, по-русски, без кавычек), отражающее суть обращений.
Для каждой темы даны ключевые слова и примеры комментариев. Ответ строго JSON: {{"названия": {{"<id>": "название"}}}}.
{topics}"""


def name_topics(result: ClusterResult, texts: Sequence[str], llm, max_examples: int = 6, batch: int = 15) -> dict[int, str]:
    names: dict[int, str] = {}
    topics = list(result.topics)
    for k in range(0, len(topics), batch):
        chunk = topics[k:k + batch]
        desc = "\n\n".join(
            f"Тема {t.id}. Ключевые слова: {', '.join(t.keywords)}.\nПримеры:\n" +
            "\n".join(f"- {texts[i][:200]}" for i in t.representatives[:max_examples]) for t in chunk)
        try:
            data = llm.chat_json([{"role": "user", "content": NAME_PROMPT.format(topics=desc)}])
        except Exception:
            continue
        got = data.get("названия", data) if isinstance(data, dict) else {}
        for t in chunk:
            v = got.get(str(t.id)) if isinstance(got, dict) else None
            if isinstance(v, str) and v.strip():
                names[t.id] = v.strip().strip('"«»')
    for t in result.topics:
        if t.id in names:
            t.name = names[t.id]
    return names


SUMMARY_PROMPT = """Ниже {n} комментариев сотрудников поддержки по обращениям абонентов ({label}); всего в выборке {total} обращений.
Вопрос пользователя: «{question}».
Ответь строго JSON:
{{"темы": [{{"тема": "название", "частота": "сколько из представленных комментариев (число или доля)", "содержание": "что именно говорят, 1–3 предложения"}}],
 "выводы": ["3–5 кратких выводов по вопросу пользователя, только по фактам из комментариев"]}}
Комментарии:
{docs}"""


def llm_summary(texts: Sequence[str], llm, question: str = "о чём говорят абоненты?", label: str = "вся выборка",
                total: Optional[int] = None) -> dict:
    docs = "\n".join(f"{i + 1}. {t[:300]}" for i, t in enumerate(texts))
    data = llm.chat_json([{"role": "user", "content": SUMMARY_PROMPT.format(
        n=len(texts), label=label, total=total or len(texts), question=question, docs=docs)}])
    if not isinstance(data, dict):
        return {"темы": [], "выводы": []}
    data.setdefault("темы", [])
    data.setdefault("выводы", [])
    return data


COMPARE_PROMPT = """Сравни, что говорили абоненты в двух периодах. Вопрос пользователя: «{question}».
Период A — {label_a} ({n_a} обращений, показано {k_a}):
{docs_a}

Период B — {label_b} ({n_b} обращений, показано {k_b}):
{docs_b}

Статистика по темам (доля обращений, A → B):
{stats}

Ответь строго JSON: {{"изменилось": ["..."], "не_изменилось": ["..."], "итог": "2–3 предложения"}}. Опирайся на комментарии и статистику, не выдумывай."""


def llm_compare(texts_a: Sequence[str], texts_b: Sequence[str], label_a: str, label_b: str, n_a: int, n_b: int,
                stats: pd.DataFrame, llm, question: str = "что изменилось?", names: Optional[dict] = None) -> dict:
    def docs(ts):
        return "\n".join(f"- {t[:250]}" for t in ts)
    rows = []
    for tid, r in stats.head(12).iterrows():
        nm = (names or {}).get(tid, str(tid))
        rows.append(f"- {nm}: {r['share_a']:.1%} → {r['share_b']:.1%} ({int(r['count_a'])} → {int(r['count_b'])})")
    data = llm.chat_json([{"role": "user", "content": COMPARE_PROMPT.format(
        question=question, label_a=label_a, label_b=label_b, n_a=n_a, n_b=n_b, k_a=len(texts_a), k_b=len(texts_b),
        docs_a=docs(texts_a), docs_b=docs(texts_b), stats="\n".join(rows) or "-")}])
    return data if isinstance(data, dict) else {"изменилось": [], "не_изменилось": [], "итог": ""}


DESCRIBE_PROMPT = """Ниже {n} текстов из одной группы отзывов/обращений клиентов (группа найдена автоматически).
Частые слова группы: {keywords}.
Опиши группу для аналитика. Ответ строго JSON:
{{"название": "2–6 слов: о чём группа", "описание": "1–2 предложения: какая проблема, вопрос или похвала объединяет тексты",
 "тональность": "негатив | позитив | смешанная | нейтральная", "однородность": "высокая | средняя | низкая — насколько тексты об одном и том же",
 "выбивающиеся": [номера текстов, которые не подходят под описание]}}
Тексты:
{docs}"""


def _placeholder_name(name) -> bool:
    n = str(name).strip().lower().replace("ё", "е")
    return not n or "о чем группа" in n or "2–6 слов" in n or "2-6 слов" in n or n in {"название", "группа", "тема"}


def describe_cluster(texts: Sequence[str], llm, keywords: Sequence[str] = (), max_chars: int = 350) -> dict:
    """Описание одной группы по 10–30 текстам (ТЗ v2): название, описание, тональность, однородность и номера
    выбивающихся текстов. Однородность и выбивающиеся — самооценка модели; проверка по меткам (research/results/
    descriptions_eval.csv) показала, что с реальной долей чужих она не связана: долю нерелевантных мерить выборочным чтением."""
    docs = "\n".join(f"{i + 1}. {' '.join(t.split())[:max_chars]}" for i, t in enumerate(texts))
    prompt = DESCRIBE_PROMPT.format(n=len(texts), keywords=", ".join(keywords[:10]) or "-", docs=docs)
    data = llm.chat_json([{"role": "user", "content": prompt}])
    if isinstance(data, dict) and _placeholder_name(data.get("название", "")):
        # маленькие модели иногда копируют подсказку из шаблона («о чём группа») — одна повторная попытка
        data = llm.chat_json([{"role": "user", "content": prompt + "\n\nВ поле «название» — сама тема группы, а не слова из шаблона."}])
    if not isinstance(data, dict):
        return {}
    out = {"name": str(data.get("название", "")).strip(), "description": str(data.get("описание", "")).strip(),
           "sentiment": str(data.get("тональность", "")).strip(), "homogeneity": str(data.get("однородность", "")).strip()}
    bad = data.get("выбивающиеся", [])
    out["outliers"] = sorted({int(x) for x in bad if str(x).isdigit() and 1 <= int(x) <= len(texts)}) if isinstance(bad, list) else []
    return out


# ---------- отчёт ----------

def topics_table(summaries: Sequence[TopicSummary]) -> pd.DataFrame:
    return pd.DataFrame([{
        "тема": s.name, "обращений": s.count, "доля": round(s.share, 3), "ключевые слова": ", ".join(s.keywords[:6]),
        "динамика": s.trend_text(), "содержание": s.content or " | ".join(q for _, q in s.quotes[:2]),
    } for s in summaries])


def report_markdown(summaries: Sequence[TopicSummary], findings: Sequence[Finding], names: Optional[dict] = None,
                    title: str = "Темы обращений", extra: Sequence[str] = ()) -> str:
    lines = [f"# {title}", ""]
    if findings:
        lines.append("## Что изменилось")
        for f in findings:
            lines.append(f"- {f.describe((names or {}).get(f.topic))}")
        lines.append("")
    lines.append("## Темы")
    lines.append("| тема | обращений | доля | ключевые слова | содержание |")
    lines.append("|---|---:|---:|---|---|")
    for s in summaries:
        content = s.content or "; ".join(f"«{q}»" for _, q in s.quotes[:2])
        lines.append(f"| {s.name} | {s.count} | {s.share:.1%} | {', '.join(s.keywords[:5])} | {content} |")
    for e in extra:
        lines.extend(["", e])
    return "\n".join(lines)


def to_json(summaries: Sequence[TopicSummary]) -> str:
    return json.dumps([{"id": s.id, "name": s.name, "count": s.count, "share": s.share, "keywords": s.keywords,
                        "quotes": s.quotes, "by": s.by, "trend": s.trend_text(), "content": s.content} for s in summaries],
                      ensure_ascii=False, indent=1)
