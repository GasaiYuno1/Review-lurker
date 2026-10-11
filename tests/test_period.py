"""Период из текста промпта: правила и LLM-вариант с заглушкой."""
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lurker.llm import FakeLLM  # noqa: E402
from lurker.period import Period, describe, llm_periods, parse_periods  # noqa: E402

REF = date(2026, 9, 7)


def spans(text):
    return [(p.start.isoformat(), p.end.isoformat()) for p in parse_periods(text, REF)]


class ParseTests(unittest.TestCase):
    def test_months_and_comparison(self):
        self.assertEqual(spans("сравни, что говорили про приложение в январе и в июле"),
                         [("2026-01-01", "2026-01-31"), ("2026-07-01", "2026-07-31")])
        self.assertEqual(spans("в мае"), [("2026-05-01", "2026-05-31")])
        self.assertEqual(spans("за октябрь"), [("2025-10-01", "2025-10-31")])  # октябрь ещё не наступил → прошлый год
        self.assertEqual(spans("за июль 2025"), [("2025-07-01", "2025-07-31")])
        self.assertEqual(parse_periods("в январе", REF)[0].label, "январь 2026")

    def test_relative(self):
        self.assertEqual(spans("покажи тренд негатива за последние 3 месяца"), [("2026-06-08", "2026-09-07")])
        self.assertEqual(spans("за последние полгода"), [("2026-03-08", "2026-09-07")])
        self.assertEqual(spans("за последние 2 недели"), [("2026-08-25", "2026-09-07")])
        self.assertEqual(spans("за последнюю неделю"), [("2026-09-01", "2026-09-07")])
        self.assertEqual(spans("за последние 10 дней"), [("2026-08-29", "2026-09-07")])
        self.assertEqual(spans("за прошлый месяц"), [("2026-08-01", "2026-08-31")])
        self.assertEqual(spans("в этом месяце"), [("2026-09-01", "2026-09-30")])
        self.assertEqual(spans("за прошлый год"), [("2025-01-01", "2025-12-31")])
        self.assertEqual(spans("как изменилась динамика домофона за полгода"), [("2026-03-08", "2026-09-07")])
        self.assertEqual(spans("за год"), [("2025-09-08", "2026-09-07")])
        self.assertEqual(spans("за 10 дней"), [("2026-08-29", "2026-09-07")])
        self.assertEqual(spans("за две недели"), [("2026-08-25", "2026-09-07")])
        self.assertEqual(spans("за 2025 год"), [("2025-01-01", "2025-12-31")])  # год с числом — не «за N лет»

    def test_ranges_and_dates(self):
        self.assertEqual(spans("что было с марта по май?"), [("2026-03-01", "2026-05-31")])
        self.assertEqual(spans("март-апрель 2026"), [("2026-03-01", "2026-04-30")])
        self.assertEqual(spans("с 1 марта по 15 апреля 2026"), [("2026-03-01", "2026-04-15")])
        self.assertEqual(spans("01.02.2026 - 28.02.2026 негатив"), [("2026-02-01", "2026-02-28")])
        self.assertEqual(spans("с ноября по февраль"), [("2025-11-01", "2026-02-28")])
        self.assertEqual(spans("в 1 квартале 2026 и во втором квартале"), [("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30")])
        self.assertEqual(spans("за 2025 год"), [("2025-01-01", "2025-12-31")])

    def test_none(self):
        self.assertEqual(spans("что говорят про скорость"), [])
        self.assertEqual(spans("майские жалобы на маршрутизатор"), [])  # «май» внутри слова не считается
        self.assertEqual(describe([]), "весь период")

    def test_llm_variant(self):
        llm = FakeLLM('{"periods": [{"start": "2026-01-01", "end": "2026-01-31", "label": "январь"}, {"start": "bad"}], "question": "x"}')
        ps = llm_periods("в январе", REF, llm)
        self.assertEqual(ps, [Period(date(2026, 1, 1), date(2026, 1, 31), "январь")])
        self.assertTrue(ps[0].contains(date(2026, 1, 15)))
        self.assertFalse(ps[0].contains(date(2026, 2, 1)))


if __name__ == "__main__":
    unittest.main()
