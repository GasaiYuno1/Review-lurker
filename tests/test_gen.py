"""Генератор тестовых данных: структура, детерминизм, «грязные» SMM-комментарии, заложенные тренды, файлы."""
import csv
import re
import sys
import tempfile
import unittest
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_testdata as g  # noqa: E402


class GenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.labels = g.generate(3000, 7, date(2026, 1, 1), date(2026, 9, 7))

    def month_counts(self, topic):
        c = Counter()
        for r, l in zip(self.rows, self.labels):
            if l["topic"] == topic:
                c[r["Дата создания"].strftime("%m")] += 1
        return c

    def test_shape_and_ids(self):
        self.assertEqual(len(self.rows), 3030)  # 3000 + 1% дублей
        self.assertEqual([r["ID"] for r in self.rows], [l["ID"] for l in self.labels])
        self.assertEqual(list(self.rows[0]), ["ID", "Дата создания", "Отдел", "Сотрудник", "Сервис", "Подкатегория", "Комментарий"])
        dates = [r["Дата создания"] for r in self.rows]
        self.assertEqual(dates, sorted(dates))
        self.assertTrue(all(date(2026, 1, 1) <= d.date() <= date(2026, 9, 10) for d in dates))

    def test_deterministic(self):
        rows2, labels2 = g.generate(3000, 7, date(2026, 1, 1), date(2026, 9, 7))
        self.assertEqual(self.rows, rows2)
        self.assertEqual(self.labels, labels2)

    def test_departments_and_styles(self):
        depts = Counter(r["Отдел"] for r in self.rows)
        self.assertEqual(set(depts), {"Колл-центр", "Техподдержка", "SMM", "Офис"})
        self.assertGreater(depts["Колл-центр"], depts["Техподдержка"] > 0)
        tech = [r for r in self.rows if r["Отдел"] == "Техподдержка"]
        self.assertTrue(all(g.TOPICS_BY_KEY[l["topic"]].service in g.TECH_SERVICES
                            for r, l in zip(self.rows, self.labels) if r["Отдел"] == "Техподдержка"))
        self.assertGreater(sum("Заявка №" in r["Комментарий"] for r in tech) / len(tech), 0.4)

    def test_smm_is_dirty(self):
        smm = [r["Комментарий"] for r in self.rows if r["Отдел"] == "SMM" and len(r["Комментарий"]) > 5]
        self.assertGreater(len(smm), 200)
        links = sum("http" in t for t in smm) / len(smm)
        phones = sum(bool(re.search(r"(\+7|8)[\d ()-]{9,}", t)) for t in smm) / len(smm)
        summary = sum(bool(re.search(r"(Итог|Саммари|Резюме|саммари):", t)) for t in smm) / len(smm)
        self.assertGreater(links, 0.7)
        self.assertGreater(phones, 0.6)
        self.assertGreater(summary, 0.9)

    def test_noise(self):
        empty = sum(r["Комментарий"] in ("", "-", ".", "см. предыдущий") for r in self.rows)
        self.assertTrue(20 <= empty <= 90, empty)
        self.assertEqual(sum(1 for l in self.labels if l["dup"] == 1), 30)
        wrong = sum(1 for r, l in zip(self.rows, self.labels) if r["Сервис"] != l["service_true"])
        self.assertTrue(100 <= wrong <= 330, wrong)
        self.assertGreater(sum(1 for l in self.labels if l["topic2"]), 30)

    def test_trend_spike_march(self):
        c = self.month_counts("домофон_приложение_не_приходит_звонок")
        base = (c["01"] + c["02"] + c["04"] + c["05"]) / 4
        self.assertGreater(c["03"], 2.5 * base, dict(c))

    def test_trend_threats_grow(self):
        c = self.month_counts("оскорбления_угрозы")
        first_half = sum(c[m] for m in ("01", "02", "03", "04", "05", "06")) / 6
        self.assertGreater(c["08"], 2 * first_half, dict(c))
        self.assertTrue(all(g.TOPICS_BY_KEY[l["topic"]].threat == bool(l["threat"]) for l in self.labels))

    def test_trend_new_topic_and_ramp(self):
        c = self.month_counts("прил_после_обновления_ошибки")
        self.assertEqual(sum(c[m] for m in ("01", "02", "03", "04", "05", "06", "07")), 0)
        self.assertGreater(c["08"], 3)
        c = self.month_counts("прил_оплата_не_проходит")
        self.assertGreater(c["08"], 2.5 * (c["02"] + c["03"] + c["04"] + c["05"]) / 4, dict(c))

    def test_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sub" / "t"
            self.assertEqual(g.main(["--rows", "200", "--seed", "3", "--out", str(out), "--quiet"]), 0)
            self.assertTrue(out.with_suffix(".csv").is_file())
            self.assertTrue(out.with_suffix(".xlsx").is_file())
            with open(out.with_suffix(".csv"), encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f, delimiter=";"))
            with open(out.with_name("t_labels.csv"), encoding="utf-8", newline="") as f:
                labels = list(csv.DictReader(f))
            self.assertEqual(len(rows), 202)
            self.assertEqual([r["ID"] for r in rows], [l["ID"] for l in labels])
            self.assertRegex(rows[0]["Дата создания"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")


if __name__ == "__main__":
    unittest.main()
