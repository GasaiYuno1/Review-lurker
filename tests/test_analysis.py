"""Эмбеддинги, кластеры, динамика, предклассификация, саммари, ассистент и пайплайн целиком на сгенерированных данных."""
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import gen_testdata as g  # noqa: E402
from lurker.clean import RuleCleaner, clean_texts  # noqa: E402
from lurker.cluster import assign_new, cluster_embeddings, evaluate, reduce_topics  # noqa: E402
from lurker.config import Config  # noqa: E402
from lurker.dynamics import apply_filters, compare_periods, detect_changes, timeline  # noqa: E402
from lurker.embed import HybridEmbedder, NavecEmbedder, TfidfEmbedder, get_embedder  # noqa: E402
from lurker.llm import FakeLLM  # noqa: E402
from lurker.period import Period  # noqa: E402
from lurker.pipeline import load, run_analysis, save  # noqa: E402
from lurker.preclass import (classify, classify_llm, evaluate_labels, label_prototypes, llm_criteria,  # noqa: E402
                             parse_criteria)
from lurker.summarize import llm_compare, llm_summary, name_topics, report_markdown, sample_texts, summarize_topics, topics_table  # noqa: E402


def make_df(rows=400, seed=3):
    r, labels = g.generate(rows, seed, date(2026, 1, 1), date(2026, 9, 7))
    df = pd.DataFrame(r).rename(columns={"Дата создания": "date", "Отдел": "department", "Сотрудник": "author",
                                        "Сервис": "service", "Подкатегория": "subservice", "Комментарий": "text", "ID": "id"})
    df["id"] = df["id"].astype(str)
    df["topic_true"] = [l["topic"] for l in labels]
    df["clean"] = clean_texts(df["text"].tolist(), RuleCleaner(ner=False))
    return df.reset_index(drop=True)


class EmbedTests(unittest.TestCase):
    def test_backends(self):
        texts = ["нет интернета", "интернет не работает", "приставка зависает", "тв не показывает", "заказать ключи"] * 4
        for emb in (TfidfEmbedder(dim=8), NavecEmbedder(), HybridEmbedder(dim=8)):
            E = emb.fit(texts).encode(texts)
            self.assertEqual(E.shape[0], len(texts))
            np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4)
            q = emb.encode(["интернет отвалился"])[0]
            sims = E @ q
            self.assertGreater(sims[0], sims[4], emb.name)  # про интернет ближе, чем про ключи
        self.assertEqual(get_embedder("tfidf").name, "tfidf")
        with self.assertRaises(ValueError):
            get_embedder("nope")


class ClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_df(600, seed=11)
        m = cls.df["clean"].str.len() > 3
        cls.texts = cls.df.loc[m, "clean"].tolist()
        cls.true = cls.df.loc[m, "topic_true"].tolist()
        cls.E = HybridEmbedder(dim=100).fit(cls.texts).encode(cls.texts)

    def test_umap_hdbscan_quality(self):
        res = cluster_embeddings(self.E, self.texts, method="umap_hdbscan")
        ev = evaluate(res.labels, self.true)
        self.assertGreater(ev["n_clusters"], 8)
        self.assertGreater(ev["purity"], 0.65, ev)
        self.assertGreater(ev["nmi"], 0.6, ev)
        self.assertEqual(len(res.labels), len(self.texts))
        top = res.topics[0]
        self.assertEqual(top.size, max(t.size for t in res.topics))
        self.assertTrue(top.keywords)
        self.assertTrue(all(res.labels[i] == top.id for i in top.representatives))
        new = assign_new(self.E[:5], res)
        self.assertEqual(list(new), list(res.labels[:5]))

    def test_reduce_and_agglomerative(self):
        res = cluster_embeddings(self.E, self.texts, method="umap_hdbscan", max_topics=6)
        self.assertLessEqual(len(res.topics), 6)
        labels = reduce_topics(self.E, res.labels, merge_threshold=0.99)
        self.assertEqual(len(set(labels[labels >= 0])), len(res.topics))
        res2 = cluster_embeddings(self.E, self.texts, method="agglomerative", distance_threshold=0.6)
        self.assertGreater(len(res2.topics), 1)

    def test_tiny_input(self):
        res = cluster_embeddings(self.E[:5], self.texts[:5])
        self.assertEqual(res.method, "single")
        self.assertEqual(len(res.topics), 1)


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        days = [pd.Timestamp("2026-01-05") + pd.Timedelta(days=i) for i in range(180)]
        rows = []
        for i, d in enumerate(days):
            rows += [{"date": d, "topic": "a"}] * 5 + [{"date": d, "topic": "b"}] * 3
            if 60 <= i < 67:
                rows += [{"date": d, "topic": "b"}] * 10          # всплеск b в марте
            if i >= 140:
                rows += [{"date": d, "topic": "c"}] * 4           # новая тема c
            if i >= 120:
                rows += [{"date": d, "topic": "a"}] * 6           # рост a
        self.df = pd.DataFrame(rows)
        self.df["service"] = "x"

    def test_timeline_and_findings(self):
        tl = timeline(self.df, "topic", "M")
        self.assertEqual(list(tl.columns), ["a", "b", "c"])
        self.assertEqual(int(tl["a"].iloc[0]), 5 * len(tl.index[0:1]) * 27)  # январь: 27 дней с 5 января
        kinds = {(f.kind, f.topic) for f in detect_changes(timeline(self.df, "topic", "W"))}
        self.assertIn(("spike", "b"), kinds)
        self.assertIn(("new", "c"), kinds)
        self.assertIn(("growth", "a"), kinds)
        sh = timeline(self.df, "topic", "W", shares=True)
        np.testing.assert_allclose(sh.sum(axis=1), 1.0)
        self.assertTrue(timeline(self.df, "topic", "W", filters={"service": "нет"}).empty)

    def test_compare_and_filters(self):
        a, b = Period(date(2026, 1, 5), date(2026, 1, 31), "янв"), Period(date(2026, 6, 1), date(2026, 7, 3), "июн")
        cmp_ = compare_periods(self.df, a, b, "topic")
        self.assertEqual(cmp_.attrs["labels"], ("янв", "июн"))
        self.assertGreater(cmp_.loc["c", "share_b"], 0)
        self.assertEqual(cmp_.loc["c", "count_a"], 0)
        self.assertEqual(len(apply_filters(self.df, periods=[a])), 27 * 8)


class PreclassTests(unittest.TestCase):
    def test_parse_and_classify(self):
        crit = parse_criteria("цифровая: приложение и звонки в приложении | примеры: не приходит звонок в приложение; дверь из приложения | слова: приложение\n"
                              "аналоговая: домофон на подъезде, трубка, ключи | примеры: трубка не работает; заказать ключи | слова: трубка, ключ, панель")
        self.assertEqual(crit.names, ["цифровая", "аналоговая"])
        self.assertEqual(crit.labels[1].keywords, ["трубка", "ключ", "панель"])
        texts = ["не приходит звонок с домофона в приложение", "кнопка открыть в приложении не работает", "не работает трубка домофона в квартире",
                 "хочет заказать ключи от домофона", "домофон на подъезде не открывает ключом", "приложение пишет ошибка при открытии двери"] * 3
        emb = HybridEmbedder(dim=8).fit(texts)
        E = emb.encode(texts)
        protos = label_prototypes(crit, emb, texts, E)
        pred, S = classify(E, crit, protos, threshold=0.0, texts=texts)
        self.assertEqual(pred[:6], ["цифровая", "цифровая", "аналоговая", "аналоговая", "аналоговая", "цифровая"])
        self.assertEqual(S.shape, (18, 2))
        pred_hi, _ = classify(E, crit, protos, threshold=0.99, texts=texts, keyword_bonus=0)
        self.assertTrue(all(p == "прочее" for p in pred_hi))
        ev = evaluate_labels(pred[:6], ["цифровая"] * 2 + ["аналоговая"] * 3 + ["цифровая"])
        self.assertEqual(ev["accuracy"], 1.0)

    def test_llm_paths(self):
        crit = llm_criteria("нас оскорбляют?", FakeLLM('{"labels": [{"name": "угрозы", "description": "d", "examples": ["x"], "keywords": ["Угрожать"]}, {"bad": 1}]}'), ["a"])
        self.assertEqual(crit.names, ["угрозы"])
        self.assertEqual(crit.labels[0].keywords, ["угрожать"])
        out = classify_llm(["a", "b", "c"], crit, FakeLLM('{"1": "угрозы", "2": "неизвестно", "3": "прочее"}'), batch_size=10)
        self.assertEqual(out, ["угрозы", "прочее", "прочее"])


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        df = make_df(500, seed=5)
        cls.an = run_analysis(df, Config(), freq="M", cache_dir=cls.tmp.name, log=lambda s: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_analysis_structure(self):
        an = self.an
        self.assertEqual(len(an.E), len(an.df))
        for col in ("clean", "topic", "topic_name"):
            self.assertIn(col, an.df.columns)
        self.assertGreater(len(an.clusters.topics), 3)
        self.assertEqual(an.df["topic"].min(), -1 if an.clusters.n_outliers or (an.df["clean"].str.len() <= 3).any() else 0)
        self.assertTrue(all(s.name for s in an.summaries))
        self.assertEqual(an.topic_name(-1), "прочее")
        tl = an.timeline()
        self.assertFalse(tl.empty)
        self.assertTrue(an.params["cleaner"].startswith("rules"))
        table = topics_table(an.summaries)
        self.assertEqual(len(table), len(an.summaries))
        md = report_markdown(an.summaries, an.findings, an.names)
        self.assertIn("## Темы", md)
        # представители — индексы в df и лежат в своей теме
        for tp in an.clusters.topics:
            for i in tp.representatives:
                self.assertEqual(an.df.loc[i, "topic"], tp.id)

    def test_cache_hit_is_fast_and_same(self):
        import time
        t = time.time()
        an2 = run_analysis(self.an.df.drop(columns=["clean", "topic", "topic_name"]), Config(), freq="M", cache_dir=self.tmp.name)
        self.assertLess(time.time() - t, 15)
        np.testing.assert_array_equal(an2.df["topic"].to_numpy(), self.an.df["topic"].to_numpy())

    def test_save_load(self):
        p = Path(self.tmp.name) / "an.pkl"
        save(self.an, p)
        an = load(p)
        self.assertEqual(len(an.df), len(self.an.df))
        self.assertEqual(an.names, self.an.names)

    def test_llm_summaries_with_fake(self):
        an = self.an
        names = name_topics(an.clusters, an.df["clean"].tolist(), FakeLLM(lambda m: '{"названия": {"0": "Первая тема"}}'))
        self.assertEqual(names.get(0), "Первая тема")
        self.assertEqual(an.clusters.topics[0].name, "Первая тема")
        s = llm_summary(sample_texts(an.df, 5), FakeLLM('{"темы": [{"тема": "т", "частота": "5", "содержание": "с"}], "выводы": ["в"]}'), "вопрос")
        self.assertEqual(s["выводы"], ["в"])
        stats = compare_periods(an.df, Period(date(2026, 1, 1), date(2026, 3, 31), "A"), Period(date(2026, 6, 1), date(2026, 9, 7), "B"), "topic")
        c = llm_compare(["a"], ["b"], "A", "B", 10, 12, stats, FakeLLM('{"изменилось": ["x"], "не_изменилось": [], "итог": "и"}'))
        self.assertEqual(c["итог"], "и")
        self.assertLessEqual(len(sample_texts(an.df, 7)), 7)

    def test_preclass_in_pipeline(self):
        crit = parse_criteria("угрозы: оскорбления и угрозы сотрудникам | примеры: угрожал физической расправой; обзывал мошенниками | слова: угрожать, оскорблять, мошенник")
        an = run_analysis(self.an.df.drop(columns=["clean", "topic", "topic_name"]), Config(), freq="M", cache_dir=self.tmp.name, criteria=crit)
        self.assertIn("label", an.df.columns)
        hit = an.df[an.df["label"] == "угрозы"]
        self.assertGreater(len(hit), 0)
        self.assertGreater((hit["topic_true"] == "оскорбления_угрозы").mean(), 0.7)


if __name__ == "__main__":
    unittest.main()
