"""Сценарий «вопрос к данным» (подпакет lurker.service, отделяемый вместе с ui/): предмет запроса, пресеты, отбор по смыслу,
динамика понятия, тематики внутри выборки, ассистент. Удаляется вместе с src/lurker/service и ui/."""
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_analysis import make_df  # noqa: E402
from lurker.service.assistant import answer, retrieve  # noqa: E402
from lurker.service.concept import build_concept, match_presets, score, subject_of  # noqa: E402
from lurker.config import Config  # noqa: E402
from lurker.llm import FakeLLM  # noqa: E402
from lurker.pipeline import run_analysis  # noqa: E402
from lurker.preclass import parse_criteria  # noqa: E402
from lurker.service.query import borderline, run_query  # noqa: E402

REF = date(2026, 9, 7)


class SubjectTests(unittest.TestCase):
    def test_subject_and_periods(self):
        s, p, core = subject_of("сравни, что говорили про мобильное приложение в январе и в июле", REF)
        self.assertEqual(core, ["мобильный", "приложение"])
        self.assertEqual([x.label for x in p], ["январь 2026", "июль 2026"])
        s, p, core = subject_of("как часто нас оскорбляли и угрожали физической расправой за последние 2 месяца", REF)
        self.assertEqual(core, ["оскорблять", "угрожать", "физический", "расправа"])
        self.assertEqual(len(p), 1)
        s, p, core = subject_of("что изменилось за последний месяц", REF)
        self.assertEqual((s, core), ("", []))

    def test_presets(self):
        self.assertEqual(match_presets("нас стали чаще хуесосить?", ["хуесосить"]), ["оскорбления", "негатив"])
        self.assertEqual(match_presets("благодарности", ["благодарность"]), ["благодарность"])
        self.assertEqual(match_presets("домофон в приложении", ["домофон", "приложение"]), [])


class QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.df = make_df(1200, seed=21)
        cls.an = run_analysis(cls.df.drop(columns=["clean"]), Config(), freq="M", cache_dir=cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_threats_selected_precisely(self):
        an = self.an
        c, _ = build_concept("как часто нас оскорбляли и угрожали физической расправой", an.embedder, an.df["clean"].tolist(), an.E, REF)
        self.assertEqual(c.presets, ["оскорбления"])
        s = score(c, an.df["clean"].tolist(), an.E)
        truth = (an.df["topic_true"] == "оскорбления_угрозы").to_numpy()
        pred = s >= 0.5
        precision = (pred & truth).sum() / max(pred.sum(), 1)
        recall = (pred & truth).sum() / max(truth.sum(), 1)
        self.assertGreater(precision, 0.8, (precision, recall))
        self.assertGreater(recall, 0.5, (precision, recall))

    def test_run_query_structure_and_dynamics(self):
        q = run_query(self.an, "как часто нас оскорбляли и угрожали физической расправой за последние 2 месяца", threshold=0.5, freq="M")
        self.assertEqual(len(q.periods), 1)
        self.assertEqual(len(q.mask), len(self.an.df))
        self.assertTrue(q.in_scope.sum() < len(self.an.df))
        self.assertTrue((q.mask <= q.in_scope).all())
        self.assertEqual(list(q.series.columns), ["count", "total", "share"])
        self.assertGreater(q.n_relevant, 0)
        self.assertIn("Релевантных обращений", q.text)
        # рост угроз ×4 за последние 2 месяца заложен в данные
        s = q.series["count"]
        self.assertGreater(s.iloc[-3:-1].sum(), s.iloc[:2].sum())
        near = borderline(q, self.an.df)
        self.assertTrue(set(near.columns) >= {"date", "clean", "score"})

    def test_compare_and_subtopics(self):
        q = run_query(self.an, "сравни, что говорили про мобильное приложение в январе и в июле", threshold=0.5, freq="M")
        self.assertIsNotNone(q.compare)
        self.assertEqual(list(q.compare.index), ["январь 2026", "июль 2026"])
        self.assertIn("релевантных", q.compare.columns)
        self.assertIn("Сравнение", q.text)
        if q.n_relevant >= 30:
            self.assertIsNotNone(q.subtopics)
            self.assertTrue(q.subsummaries)
            for tp in q.subtopics.topics:
                for i in tp.representatives:
                    self.assertTrue(q.mask[i])

    def test_no_subject_means_everything(self):
        q = run_query(self.an, "что изменилось за последний месяц", freq="W")
        self.assertEqual(q.concepts, [])
        nonempty = (self.an.df["clean"].str.len() > 3).to_numpy()
        self.assertEqual(q.n_relevant, int((q.in_scope & nonempty).sum()))
        self.assertGreater(q.n_scope, 0)

    def test_criteria_and_llm_criteria(self):
        crit = parse_criteria("угрозы: оскорбления и угрозы | примеры: угрожал расправой; обзывал мошенниками | слова: угрожать, оскорблять, мошенник\n"
                              "благодарность: спасибо, довольны | примеры: благодарит мастера | слова: благодарить, спасибо")
        q = run_query(self.an, "что у нас с угрозами и благодарностями", threshold=0.5, freq="M", criteria=crit)
        self.assertEqual([c.name for c in q.concepts], ["угрозы", "благодарность"])
        self.assertIn("count_угрозы", q.series.columns)
        self.assertTrue(set(q.labels.unique()) <= {"угрозы", "благодарность", "прочее"})
        fake = FakeLLM('{"labels": [{"name": "оскорбления", "description": "оскорбления и угрозы", "examples": ["угрожал расправой"], "keywords": ["угрожать", "оскорблять"]}]}')
        q2 = run_query(self.an, "нас оскорбляют?", threshold=0.5, freq="M", llm=fake, llm_criteria_augment=True)
        self.assertEqual([c.name for c in q2.concepts], ["оскорбляют"])       # понятие из промпта остаётся главным
        self.assertIn("угрожать", q2.concepts[0].preset_keywords)             # критерии LLM дополнили его словами
        self.assertIn("угрожал расправой", q2.concepts[0].examples)           # и примерами
        self.assertGreaterEqual(fake.calls, 1)

    def test_assistant_plain_and_llm(self):
        an = self.an
        ans = answer("что говорили про домофон в марте?", an.df, an.E, an.embedder, freq="M", names=an.names)
        self.assertEqual(len(ans.periods), 1)
        self.assertEqual(ans.periods[0].start, date(2026, 3, 1))
        self.assertIn("Период:", ans.text)
        self.assertTrue(ans.quotes)
        self.assertTrue(all("домофон" in q["text"] or q["score"] > 0 for q in ans.quotes))
        ans2 = answer("что говорят про приставку", an.df, an.E, an.embedder, llm=FakeLLM("ответ модели"), names=an.names)
        self.assertEqual(ans2.text, "ответ модели")
        self.assertEqual(ans2.periods, [])
        r = retrieve("приставка зависает", an.df, an.E, an.embedder, k=3)
        self.assertEqual(len(r), 3)
        self.assertIn("приставк", r.iloc[0]["clean"])

    def test_llm_period_fallback_only_when_rules_find_nothing(self):
        def reply(msgs):
            if "periods" in msgs[-1]["content"]:
                return '{"periods": [{"start": "2026-02-01", "end": "2026-02-28", "label": "февраль"}]}'
            return '{"labels": []}'
        fake = FakeLLM(reply)
        q = run_query(self.an, "домофон в приложении в марте", threshold=0.5, freq="M", llm=fake, criteria=parse_criteria("домофон: домофон | слова: домофон"))
        self.assertEqual(q.periods[0].label, "март 2026")          # правила нашли — LLM не спрашиваем
        q = run_query(self.an, "домофон в приложении во время акции", threshold=0.5, freq="M", llm=fake, criteria=parse_criteria("домофон: домофон | слова: домофон"))
        self.assertEqual([p.label for p in q.periods], ["февраль"])   # правила не нашли — взяли у LLM
        q = run_query(self.an, "домофон в приложении во время акции", threshold=0.5, freq="M", llm=fake, criteria=parse_criteria("домофон: домофон | слова: домофон"), llm_periods_fallback=False)
        self.assertEqual(q.periods, [])


if __name__ == "__main__":
    unittest.main()
