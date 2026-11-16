"""Период из текста промпта (ТЗ п. 6.2): «сравни январь и июль», «за последние 3 месяца», «с марта по май».

Два разборщика с одинаковым результатом: правила (без LLM, здесь) и LLM (`llm_periods`, промпт → тот же JSON).
Относительные периоды считаются от `ref` — максимальной даты в данных, а не от «сегодня»:
файл может быть выгружен давно. Возвращается список Period; два и больше — это сравнение периодов.
"""
from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Sequence

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
MONTH_RE = r"(январ[ьяе]|феврал[ьяе]|март[ае]?|апрел[ьяе]|ма[йяе]|июн[ьяе]|июл[ьяе]|август[ае]?|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе])"
YEAR_RE = r"(?:\s*(?:в\s+)?(\d{4})(?:\s*г\.?|\s*году)?)?"
UNITS = {"дн": "day", "недел": "week", "месяц": "month", "мес": "month", "кварт": "quarter", "полгод": "half", "полугод": "half", "год": "year", "лет": "year"}
WORD_NUM = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
            "девять": 9, "десять": 10, "двенадцать": 12, "пару": 2, "пара": 2, "несколько": 3}
QUARTER_WORDS = {"перв": 1, "1": 1, "втор": 2, "2": 2, "трет": 3, "3": 3, "четверт": 4, "4": 4}


@dataclass(frozen=True)
class Period:
    start: date
    end: date          # включительно
    label: str

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end

    def to_json(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "label": self.label}


def _month_num(word: str) -> int:
    w = word.lower()
    for stem, n in MONTHS.items():
        if w.startswith(stem) and (stem != "ма" or w in ("май", "мая", "мае")):
            return n
    raise ValueError(word)


RU_MONTHS = ["", "январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]


def _month_period(m: int, y: int) -> Period:
    return Period(date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1]), f"{RU_MONTHS[m]} {y}")


def _shift_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _year_for(month: int, explicit: Optional[str], ref: date) -> int:
    if explicit:
        return int(explicit)
    # без года: последний такой месяц, не позже ref
    return ref.year if month <= ref.month else ref.year - 1


def parse_periods(text: str, ref: date) -> list[Period]:
    """Список периодов из текста; пусто — период не указан (берутся все данные)."""
    return _parse(text, ref)[0]


def strip_periods(text: str, ref: date) -> tuple[str, list[Period]]:
    """Текст без упоминаний периода (в нижнем регистре) и сами периоды — для выделения предмета запроса."""
    periods, spans, t = _parse(text, ref, with_spans=True)
    out, pos = [], 0
    for a, b in sorted(spans):
        out.append(t[pos:a])
        pos = b
    out.append(t[pos:])
    rest = re.sub(r"\s+", " ", "".join(out)).strip()
    return rest, periods


def _parse(text: str, ref: date, with_spans: bool = False):
    t = " " + text.lower().replace("ё", "е") + " "
    periods: list[Period] = []
    used: list[tuple[int, int]] = []

    def take(m: re.Match, p: Period | Sequence[Period]):
        for u in used:
            if not (m.end() <= u[0] or m.start() >= u[1]):
                return
        used.append((m.start(), m.end()))
        for x in ([p] if isinstance(p, Period) else p):
            if x not in periods:
                periods.append(x)

    # 1. явные даты: с 01.03.2026 по 15.04.2026 / 01.03.2026 - 15.04.2026 / с 1 марта по 15 апреля
    for m in re.finditer(r"(\d{1,2})[./](\d{1,2})[./](\d{4})\s*(?:-|—|по|до)\s*(\d{1,2})[./](\d{1,2})[./](\d{4})", t):
        d1, m1, y1, d2, m2, y2 = map(int, m.groups())
        take(m, Period(date(y1, m1, d1), date(y2, m2, d2), f"{d1:02d}.{m1:02d}.{y1}–{d2:02d}.{m2:02d}.{y2}"))
    for m in re.finditer(r"(?:с|от)\s+(\d{1,2})\s+" + MONTH_RE + YEAR_RE + r"\s+(?:по|до)\s+(\d{1,2})\s+" + MONTH_RE + YEAR_RE, t):
        d1, mo1, y1, d2, mo2, y2 = m.groups()
        mn2 = _month_num(mo2)
        yy2 = _year_for(mn2, y2, ref)
        mn1 = _month_num(mo1)
        yy1 = int(y1) if y1 else (yy2 if mn1 <= mn2 else yy2 - 1)
        take(m, Period(date(yy1, mn1, int(d1)), date(yy2, mn2, int(d2)), f"{int(d1)} {mo1} {yy1} – {int(d2)} {mo2} {yy2}"))

    # 2. диапазон месяцев: с марта по май, март-май, с января 2026 по март
    for m in re.finditer(r"(?:с|от)?\s*" + MONTH_RE + YEAR_RE + r"\s*(?:-|—|по|до)\s*" + MONTH_RE + YEAR_RE, t):
        mo1, y1, mo2, y2 = m.groups()
        mn1, mn2 = _month_num(mo1), _month_num(mo2)
        yy2 = _year_for(mn2, y2, ref)
        yy1 = int(y1) if y1 else (yy2 if mn1 <= mn2 else yy2 - 1)
        p1, p2 = _month_period(mn1, yy1), _month_period(mn2, yy2)
        take(m, Period(p1.start, p2.end, f"{p1.label} – {p2.label}"))

    # 3. последние N единиц / последний месяц / прошлый месяц / этот месяц
    num = r"(\d+|" + "|".join(WORD_NUM) + r")?"
    for m in re.finditer(r"последн(?:ие|ий|юю|яя|ее|их)\s+" + num + r"\s*(дн|недел|месяц|мес|кварт|полгод|полугод|год|лет)\w*", t):
        n_raw, unit = m.groups()
        n = int(n_raw) if n_raw and n_raw.isdigit() else WORD_NUM.get(n_raw or "", 1)
        take(m, _last(n, UNITS[unit], ref))
    for m in re.finditer(r"(прошл|предыдущ|этот|этом|текущ|нынешн)\w*\s+(недел|месяц|мес|кварт|год)\w*", t):
        which, unit = m.groups()
        take(m, _relative(which, UNITS[unit], ref))
    # «за полгода», «за год», «за месяц», «за 10 дней», «за две недели» — без слова «последние» смысл тот же
    for m in re.finditer(r"\bза\s+" + num + r"\s*(дн|недел|месяц|мес|кварт|полгод|полугод|год)\w*", t):
        n_raw, unit = m.groups()
        if n_raw and n_raw.isdigit() and int(n_raw) >= 1000:
            continue  # «за 2025 год» — это год, его разбирает правило ниже
        n = int(n_raw) if n_raw and n_raw.isdigit() else WORD_NUM.get(n_raw or "", 1)
        take(m, _last(n, UNITS[unit], ref))

    # 4. кварталы: в 1 квартале 2026, во втором квартале
    for m in re.finditer(r"(перв|втор|трет|четверт|[1-4])\w*\s+кварт\w*" + YEAR_RE, t):
        q = QUARTER_WORDS[m.group(1)[:5] if not m.group(1).isdigit() else m.group(1)]
        y = int(m.group(2)) if m.group(2) else ref.year
        take(m, Period(date(y, 3 * q - 2, 1), date(y, 3 * q, calendar.monthrange(y, 3 * q)[1]), f"{q} квартал {y}"))

    # 5. отдельные месяцы: в январе, за июль 2025, январе и июле
    for m in re.finditer(r"(?<![а-я])" + MONTH_RE + YEAR_RE + r"(?![а-я])", t):
        mn = _month_num(m.group(1))
        take(m, _month_period(mn, _year_for(mn, m.group(2), ref)))

    # 6. год: в 2025 году, за 2025
    for m in re.finditer(r"(?:в|за)\s+(\d{4})(?:\s*г\.?|\s*году?)?(?![.\d])", t):
        y = int(m.group(1))
        if 2000 <= y <= 2100:
            take(m, Period(date(y, 1, 1), date(y, 12, 31), str(y)))

    periods.sort(key=lambda p: p.start)
    if with_spans:
        return periods, used, t
    return periods, used


def _last(n: int, unit: str, ref: date) -> Period:
    if unit == "day":
        return Period(ref - timedelta(days=n - 1), ref, f"последние {n} дн.")
    if unit == "week":
        return Period(ref - timedelta(days=7 * n - 1), ref, f"последние {n} нед.")
    if unit == "month":
        return Period(_shift_months(ref, -n) + timedelta(days=1), ref, f"последние {n} мес.")
    if unit == "quarter":
        return Period(_shift_months(ref, -3 * n) + timedelta(days=1), ref, f"последние {n} кв.")
    if unit == "half":
        return Period(_shift_months(ref, -6 * n) + timedelta(days=1), ref, "последние полгода" if n == 1 else f"последние {6 * n} мес.")
    return Period(_shift_months(ref, -12 * n) + timedelta(days=1), ref, f"последние {n} г.")


def _relative(which: str, unit: str, ref: date) -> Period:
    prev = which.startswith(("прошл", "предыдущ"))
    if unit == "week":
        start = ref - timedelta(days=ref.weekday())
        if prev:
            start -= timedelta(days=7)
        return Period(start, start + timedelta(days=6), "прошлая неделя" if prev else "эта неделя")
    if unit == "month":
        y, m = (ref.year, ref.month)
        if prev:
            y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        p = _month_period(m, y)
        return Period(p.start, p.end, "прошлый месяц" if prev else "этот месяц")
    if unit == "quarter":
        q = (ref.month - 1) // 3 + 1
        y = ref.year
        if prev:
            q, y = (4, y - 1) if q == 1 else (q - 1, y)
        return Period(date(y, 3 * q - 2, 1), date(y, 3 * q, calendar.monthrange(y, 3 * q)[1]), "прошлый квартал" if prev else "этот квартал")
    y = ref.year - 1 if prev else ref.year
    return Period(date(y, 1, 1), date(y, 12, 31), "прошлый год" if prev else "этот год")


# ---------- вариант через LLM: тот же результат ----------

PERIOD_PROMPT = """Ты извлекаешь временные периоды из запроса пользователя к данным.
Сегодняшняя дата (последняя дата в данных): {ref}. Данные охватывают {data_start} – {data_end}.
Верни JSON: {{"periods": [{{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD", "label": "короткое название"}}], "question": "запрос без указания периода"}}.
Если период не указан — periods пустой. Если пользователь сравнивает два периода — верни оба. Относительные периоды («последние 3 месяца») считай от {ref}.
Запрос: {text}"""


def llm_periods(text: str, ref: date, llm, data_start: Optional[date] = None, data_end: Optional[date] = None) -> list[Period]:
    msgs = [{"role": "user", "content": PERIOD_PROMPT.format(ref=ref.isoformat(), text=text,
                                                              data_start=(data_start or ref).isoformat(),
                                                              data_end=(data_end or ref).isoformat())}]
    data = llm.chat_json(msgs)
    out = []
    for p in data.get("periods", []) if isinstance(data, dict) else []:
        try:
            out.append(Period(date.fromisoformat(p["start"]), date.fromisoformat(p["end"]), str(p.get("label") or p["start"])))
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda p: p.start)


def describe(periods: Sequence[Period]) -> str:
    return ", ".join(f"{p.label} ({p.start.isoformat()} – {p.end.isoformat()})" for p in periods) or "весь период"


if __name__ == "__main__":  # быстрая проверка: python3 -m lurker.period "сравни январь и июль"
    import sys
    print(json.dumps([p.to_json() for p in parse_periods(" ".join(sys.argv[1:]), date(2026, 9, 7))], ensure_ascii=False, indent=1))
