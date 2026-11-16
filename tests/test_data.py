"""Загрузка CSV/XLSX, автоопределение колонок, нормализация дат."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from lurker.data import ColumnMap, guess_columns, load_table, normalize, parse_dates  # noqa: E402
import gen_testdata as g  # noqa: E402


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "t"
        g.main(["--rows", "300", "--seed", "5", "--out", str(cls.out), "--quiet"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_csv_and_xlsx_agree(self):
        a = load_table(self.out.with_suffix(".csv"))
        b = load_table(self.out.with_suffix(".xlsx"))
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(list(a.columns), list(b.columns))
        self.assertEqual(a["Комментарий"].tolist(), b["Комментарий"].fillna("").astype(str).tolist())

    def test_guess_columns_by_names(self):
        df = load_table(self.out.with_suffix(".csv"))
        cm = guess_columns(df)
        self.assertEqual(cm, ColumnMap(text="Комментарий", date="Дата создания", id="ID", service="Сервис",
                                       subservice="Подкатегория", author="Сотрудник", department="Отдел"))

    def test_normalize(self):
        for suffix in (".csv", ".xlsx"):
            df = load_table(self.out.with_suffix(suffix))
            n = normalize(df, guess_columns(df))
            self.assertEqual(n["date"].isna().sum(), 0)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(n["date"]))
            self.assertEqual(n["date"].min().year, 2026)
            self.assertTrue((n["text"].str.strip() == n["text"]).all())
            self.assertIn("Колл-центр", set(n["department"]))
            self.assertEqual(len(n), len(df))

    def test_guess_columns_by_content(self):
        df = pd.DataFrame({
            "a": ["05.03.2026 10:00", "06.03.2026 11:30", "07.03.2026 09:15"],
            "b": ["короткое", "поле", "тут"],
            "c": ["длинный комментарий абонента про интернет", "ещё один длинный комментарий", "и третий тоже длинный"],
        })
        cm = guess_columns(df)
        self.assertEqual((cm.date, cm.text), ("a", "c"))
        self.assertIsNone(cm.service)

    def test_guess_columns_fails_without_date(self):
        df = pd.DataFrame({"x": ["a", "b"], "y": ["c", "d"]})
        with self.assertRaises(ValueError):
            guess_columns(df)

    def test_csv_cp1251_comma(self):
        p = Path(self.tmp.name) / "win.csv"
        p.write_bytes("Дата,Текст\n01.02.2026,привет\n02.02.2026,\"а, б\"\n".encode("cp1251"))
        df = load_table(p)
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df["Текст"].tolist(), ["привет", "а, б"])
        n = normalize(df, guess_columns(df))
        self.assertEqual(n["date"].iloc[0].month, 2)
        self.assertEqual(n["date"].iloc[0].day, 1)

    def test_parse_dates_dayfirst(self):
        s = pd.Series(["05.03.2026", "13.03.2026 18:00", "2026-03-14T10:00:00", ""])
        p = parse_dates(s)
        self.assertEqual((p.iloc[0].day, p.iloc[0].month), (5, 3))
        self.assertEqual((p.iloc[1].day, p.iloc[1].hour), (13, 18))
        self.assertEqual(p.iloc[2].day, 14)
        self.assertTrue(pd.isna(p.iloc[3]))

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            load_table(Path(self.tmp.name) / "x.json")


if __name__ == "__main__":
    unittest.main()
