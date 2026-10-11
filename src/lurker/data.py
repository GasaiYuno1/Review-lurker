"""Загрузка CSV/XLSX и маппинг колонок в стандартный вид (ТЗ п. 3: набор колонок заранее неизвестен).

Стандартные колонки после normalize(): id, date, text, service, subservice, author, department
(+ все исходные колонки сохраняются как есть, чтобы по ним можно было фильтровать).
guess_columns() подбирает маппинг по названиям (рус/англ синонимы), затем по содержимому
(дата — что парсится как дата, текст — самая длинная строковая колонка); в UI пользователь правит.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import pandas as pd

STANDARD = ("id", "date", "text", "service", "subservice", "author", "department")

# Синонимы заголовков (в нижнем регистре, подстрока). Порядок важен: первое совпадение побеждает.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "text": ("коммент", "текст", "описание", "расшифровк", "содержан", "comment", "text", "description", "body", "message"),
    "date": ("дата", "date", "created", "время", "time", "timestamp"),
    "subservice": ("подкатег", "подсервис", "подуслуг", "subcategory", "subservice", "sub_"),
    "service": ("сервис", "услуг", "категор", "тема", "service", "category", "topic", "product"),
    "department": ("отдел", "департамент", "подраздел", "department", "team", "group"),
    "author": ("сотрудник", "автор", "оператор", "менеджер", "author", "employee", "operator", "agent", "user", "owner"),
    "id": ("id", "номер", "ticket", "тикет", "№"),
}


@dataclass
class ColumnMap:
    text: str
    date: str
    id: Optional[str] = None
    service: Optional[str] = None
    subservice: Optional[str] = None
    author: Optional[str] = None
    department: Optional[str] = None

    def as_dict(self) -> dict[str, Optional[str]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------- загрузка ----------

def load_table(path: str | Path, sheet: Optional[str | int] = None) -> pd.DataFrame:
    """CSV (кодировка и разделитель определяются автоматически) или XLSX/XLS (первый лист по умолчанию)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    elif suffix in (".csv", ".txt", ".tsv"):
        raw = path.read_bytes()
        text, _enc = _decode(raw)
        sep = _sniff_sep(text[:65536])
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"неизвестный формат файла: {path.name} (ожидается CSV или XLSX)")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8?"


def _sniff_sep(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        return max(",;\t|", key=first.count) if first else ","


# ---------- маппинг ----------

def guess_columns(df: pd.DataFrame) -> ColumnMap:
    cols = list(df.columns)
    taken: set[str] = set()
    found: dict[str, Optional[str]] = {k: None for k in STANDARD}

    lowered = {c: c.lower() for c in cols}
    for role in ("text", "date", "subservice", "service", "department", "author", "id"):
        for c in cols:
            if c in taken:
                continue
            if any(s in lowered[c] for s in SYNONYMS[role]):
                found[role] = c
                taken.add(c)
                break

    if found["date"] is None:
        best, best_score = None, 0.0
        for c in cols:
            if c in taken:
                continue
            score = _date_score(df[c])
            if score > best_score:
                best, best_score = c, score
        if best is not None and best_score >= 0.8:
            found["date"] = best
            taken.add(best)

    if found["text"] is None:
        best, best_len = None, 0.0
        for c in cols:
            if c in taken:
                continue
            s = df[c].dropna().astype(str)
            if s.empty:
                continue
            mean_len = s.str.len().mean()
            if mean_len > best_len:
                best, best_len = c, mean_len
        if best is not None:
            found["text"] = best
            taken.add(best)

    if found["text"] is None or found["date"] is None:
        missing = [r for r in ("text", "date") if found[r] is None]
        raise ValueError(f"не удалось определить колонки: {', '.join(missing)}; укажи маппинг вручную")
    return ColumnMap(**found)  # type: ignore[arg-type]


def _date_score(s: pd.Series) -> float:
    sample = s.dropna().astype(str)
    sample = sample[sample.str.strip() != ""].head(200)
    if sample.empty:
        return 0.0
    parsed = parse_dates(sample)
    return float(parsed.notna().mean())


def parse_dates(s: pd.Series) -> pd.Series:
    """Даты в любом разумном формате. Сначала ISO 8601 (иначе pandas с dayfirst переставляет месяц и день в «2026-09-03»),
    затем смешанные форматы: dd.mm против mm/dd — выбирается вариант с меньшим числом NaT, при равенстве — день первым."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    s = s.astype(str).str.strip()
    nonempty = (s != "") & (s.str.lower() != "nan") & (s.str.lower() != "nat")
    iso = pd.to_datetime(s.where(nonempty), errors="coerce", format="ISO8601")
    if nonempty.any() and iso.notna().sum() >= 0.9 * nonempty.sum():
        return iso
    best = None
    for dayfirst in (True, False):
        parsed = pd.to_datetime(s.where(nonempty), errors="coerce", format="mixed", dayfirst=dayfirst)
        if best is None or parsed.notna().sum() > best.notna().sum():
            best = parsed
    return best


def normalize(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    """Возвращает копию с добавленными стандартными колонками (см. STANDARD); текст — str без краевых пробелов,
    date — datetime64 (NaT, если не распарсилось); отсутствующие роли заполняются пустой строкой / None."""
    out = df.copy()
    out["text"] = out[cmap.text].fillna("").astype(str).str.strip()
    out["date"] = parse_dates(out[cmap.date])
    for role in ("service", "subservice", "author", "department"):
        src = getattr(cmap, role)
        out[role] = out[src].fillna("").astype(str).str.strip() if src else ""
    out["id"] = out[cmap.id].astype(str) if cmap.id else [str(i) for i in range(len(out))]
    return out
