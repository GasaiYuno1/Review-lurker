"""Готовые скрипты для своих данных (research/run_comments.py, research/run_calls.py) и их кирпичики:
разбор расшифровки на реплики, поблочная новизна реплик, описание кластера LLM с переспросом названия-заглушки."""
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lurker.anomaly import doc_max, split_dialog, turn_novelty, turn_score  # noqa: E402
from lurker.llm import FakeLLM  # noqa: E402
from lurker.summarize import describe_cluster  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "research" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOPICS = {
    "домофон": ["не работает домофон в подъезде", "домофон не открывает дверь", "сломалась трубка домофона",
                "домофон молчит при звонке", "заказать ключ от домофона"],
    "интернет": ["пропадает интернет вечером", "низкая скорость интернета", "роутер горит красным и нет интернета",
                 "интернет обрывается каждые полчаса", "не работает вайфай после грозы"],
    "оплата": ["двойное списание абонплаты", "не прошёл платёж в приложении", "как оплатить без комиссии",
               "непонятная сумма в квитанции", "прошу перерасчёт за месяц без связи"],
}


class HelpersTests(unittest.TestCase):
    def test_split_dialog(self):
        t = split_dialog("Клиент: Привет, не работает домофон. Оператор: Здравствуйте! Назовите адрес.\nАбонент: Ленина 5\nМенеджер:Записал.")
        self.assertEqual([s for s, _ in t], ["cl", "op", "cl", "op"])
        self.assertEqual(t[1][1], "Здравствуйте! Назовите адрес.")
        self.assertEqual(split_dialog("Спикер 1: алло Спикер 2: да", ["спикер 1"], ["спикер 2"]), [("op", "алло"), ("cl", "да")])
        self.assertEqual(split_dialog("просто текст без меток"), [])

    def test_turn_novelty_blocks_and_doc_max(self):
        rng = np.random.default_rng(0)
        T = rng.normal(size=(400, 16)).astype(np.float32)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        doc = np.sort(rng.integers(0, 80, 400))
        topics = rng.integers(0, 3, 80)
        a, b = turn_novelty(T, doc, topics, block=2048), turn_novelty(T, doc, topics, block=7)
        self.assertLess(np.abs(a - b).max(), 1e-5)
        s = turn_score(T, doc, topics, top=1)
        for d in range(80):
            m = doc == d
            self.assertAlmostEqual(s[d], a[m].max() if m.any() else 0.0, places=5)
        self.assertTrue(np.allclose(doc_max(np.array([0.1, 0.5, 0.3]), np.array([0, 0, 1]), 3, top=2), [0.3, 0.3, 0.0]))

    def test_describe_cluster_retries_placeholder_name(self):
        replies = iter([json.dumps({"название": "О чём группа", "описание": "x"}, ensure_ascii=False),
                        json.dumps({"название": "Отели и завтраки", "описание": "Отзывы о гостиницах", "выбивающиеся": [2, 99]},
                                   ensure_ascii=False)])
        llm = FakeLLM(lambda m: next(replies))
        d = describe_cluster(["отель", "гостиница", "завтрак"], llm, ["отель"])
        self.assertEqual(llm.calls, 2)
        self.assertEqual(d["name"], "Отели и завтраки")
        self.assertEqual(d["outliers"], [2])

    def test_describe_cluster_retries_foreign_script(self):
        replies = iter([json.dumps({"название": "Отзывы о服务质量", "описание": "x"}, ensure_ascii=False),
                        json.dumps({"название": "Отзывы о качестве обслуживания", "описание": "y"}, ensure_ascii=False)])
        llm = FakeLLM(lambda m: next(replies))
        d = describe_cluster(["спасибо", "отлично"], llm)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(d["name"], "Отзывы о качестве обслуживания")


class NoveltyTests(unittest.TestCase):
    def test_unusual_and_emerging(self):
        from lurker.novelty import emerging_groups, unusual_scores
        rng = np.random.default_rng(0)
        base = rng.normal(size=(3, 32))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        n_old, n_new = 600, 12
        E = base[rng.integers(0, 3, n_old)] + 0.05 * rng.normal(size=(n_old, 32))
        new_dir = rng.normal(size=32)
        odd = rng.normal(size=(3, 32))                           # три одиночных необычных — каждый в своём направлении
        E = np.vstack([E, new_dir / np.linalg.norm(new_dir) + 0.05 * rng.normal(size=(n_new, 32)), odd])
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        texts = ["обычный текст из четырёх слов"] * len(E)
        texts[0] = "коротко"
        dates = pd.to_datetime(["2026-01-15"] * 300 + ["2026-02-15"] * 300 + ["2026-03-10"] * n_new + ["2026-02-20"] * 3)
        z = unusual_scores(E, texts)
        self.assertTrue(np.isneginf(z[0]))
        self.assertEqual(set(np.argsort(-z)[:3]), {n_old + n_new, n_old + n_new + 1, n_old + n_new + 2})
        eg = emerging_groups(E, dates)
        self.assertEqual(len(eg), 1)
        self.assertEqual(eg.iloc[0].period, "2026-03")
        self.assertEqual(sorted(eg.iloc[0].members), list(range(n_old, n_old + n_new)))


class SplitLargeTests(unittest.TestCase):
    def test_split_large_divides_only_big_cluster(self):
        from lurker.cluster import split_large
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(4, 24))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        y = np.repeat([0, 1, 2, 3], [200, 200, 60, 60])          # «мешок» из двух тем + две маленькие темы
        E = centers[y] + 0.08 * rng.normal(size=(len(y), 24))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        labels = np.where(y <= 1, 0, y - 1)                        # первая ступень склеила темы 0 и 1
        out = split_large(E, ["x"] * len(y), labels, share=0.5)
        self.assertGreaterEqual(len(set(out[y <= 1].tolist())), 2)
        self.assertEqual(len(set(out[y == 2].tolist())), 1)
        self.assertEqual(len(set(out[y == 3].tolist())), 1)


class RunScriptsTests(unittest.TestCase):
    def test_run_comments(self):
        rnd = random.Random(0)
        rows = []
        for topic, phrases in TOPICS.items():
            for i in range(30):
                rows.append({"Комментарий": f"{rnd.choice(phrases)}, {rnd.choice(phrases)}. тел 8{rnd.randint(10**9, 10**10 - 1)} "
                                            f"https://vk.com/id{rnd.randint(1, 10**6)}", "Дата": f"{1 + i % 28:02d}.0{1 + i % 3}.2026",
                             "Тема": topic})
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "t.csv"
            pd.DataFrame(rows).to_csv(src, index=False)
            llm = FakeLLM(json.dumps({"название": "Жалобы на связь", "описание": "Описание", "тональность": "негатив"}, ensure_ascii=False))
            rc = load_script("run_comments").main([str(src), "--text", "Комментарий", "--date", "Дата", "--model", "tfidf",
                                                   "--mcs", "10", "--out", tmp, "--label", "Тема"], llm=llm)
            self.assertEqual(rc, 0)
            cl = pd.read_csv(Path(tmp) / "clusters.csv")
            asg = pd.read_csv(Path(tmp) / "assignments.csv")
            self.assertGreaterEqual(len(cl), 2)
            self.assertEqual(len(asg), len(rows))
            self.assertNotIn("vk.com", " ".join(asg["clean_text"].astype(str)))
            self.assertEqual(llm.calls, len(cl))
            for f in ("manual_check.csv", "dynamics.csv", "clusters.md", "boilerplate.csv", "unusual.csv", "new_topics.csv"):
                self.assertTrue((Path(tmp) / f).exists(), f)
            met = pd.read_csv(Path(tmp) / "label_metrics.csv")
            self.assertEqual(met["label"].tolist(), ["Тема"])
            self.assertGreater(met["nmi"].iloc[0], 0.5)
            self.assertTrue(set(cl["label_top"]) <= set(TOPICS))

    def test_run_calls_finds_rude_operator(self):
        rnd = random.Random(1)
        answers = {"домофон": ["оформил заявку на ремонт домофона мастер придёт завтра", "проверил домофон на линии всё в порядке",
                               "ключ можно забрать в офисе"],
                   "интернет": ["перезагрузите роутер и проверьте кабель", "на линии авария срок устранения вечер",
                                "оформил выезд мастера по интернету"],
                   "оплата": ["сделал перерасчёт вернём на счёт", "платёж зависнет до суток потом пройдёт",
                              "квитанцию отправил на почту"]}
        rows = []
        for topic, phrases in TOPICS.items():
            for i in range(25):
                rows.append({"id": f"{topic}-{i}", "text": f"Клиент: {rnd.choice(phrases)}\nОператор: здравствуйте слушаю вас\n"
                                                          f"Клиент: {rnd.choice(phrases)}\nОператор: {rnd.choice(answers[topic])}"})
        rude = "домофон-7"
        rows[7]["text"] += "\nОператор: сами виноваты идиот отстаньте от меня со своим барахлом"
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "c.csv"
            pd.DataFrame(rows).to_csv(src, index=False)
            rc = load_script("run_calls").main([str(src), "--text", "text", "--id", "id", "--model", "tfidf", "--out", tmp, "--top", "5"])
            self.assertEqual(rc, 0)
            sc = pd.read_csv(Path(tmp) / "calls_scored.csv")
            self.assertEqual(len(sc), len(rows))
            self.assertIn(rude, sc["id"].head(3).tolist())
            self.assertIn("идиот", sc.set_index("id").loc[rude, "worst_operator_turn"])
            self.assertIn(rude, (Path(tmp) / "top.md").read_text(encoding="utf-8"))

    def test_run_calls_swapped_roles_topic_column_and_repeated_ids(self):
        """Метки говорящих перепутаны (оператор под «Клиент:»), темы — из колонки, на звонок по две строки."""
        rnd = random.Random(2)
        answers = {"домофон": ["оформил заявку на ремонт домофона мастер придёт завтра", "ключ можно забрать в офисе"],
                   "интернет": ["перезагрузите роутер и проверьте кабель", "оформил выезд мастера по интернету"],
                   "оплата": ["сделал перерасчёт вернём на счёт", "квитанцию отправил на почту"]}
        rows = []
        for topic, phrases in TOPICS.items():
            for i in range(25):
                text = (f"Клиент: здравствуйте меня зовут анна чем могу помочь\nОператор: {rnd.choice(phrases)}\n"
                        f"Клиент: {rnd.choice(answers[topic])}\nОператор: спасибо")
                rows += [{"id": f"{topic}-{i}", "text": text, "Тема": topic, "Дата": "01.09.2026"}] * 2
        rows.append({"id": "auto-1", "text": "Оператор: абонент сейчас не может ответить перезвоните позже", "Тема": "оплата",
                     "Дата": "01.09.2026"})
        rows.append({"id": "auto-2", "text": "Оператор: здравствуйте абонент пока не в сети я голосовой ассистент что передать\n"
                     "Клиент: добрый день это компания по поводу заявки", "Тема": "оплата", "Дата": "01.09.2026"})
        rows.append({"id": "mono-1", "text": "Оператор: сори зе конекшн из лост", "Тема": "оплата", "Дата": "01.09.2026"})
        rude = "оплата-3"
        for r in rows:
            if r["id"] == rude:
                r["text"] = r["text"] + "\nКлиент: сами виноваты идиот отстаньте от меня со своим барахлом"
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "c.csv"
            pd.DataFrame(rows).to_csv(src, index=False)
            rc = load_script("run_calls").main([str(src), "--text", "text", "--id", "id", "--model", "tfidf", "--out", tmp,
                                                "--roles", "auto", "--topic", "Тема", "--min-topic", "10", "--center-on", "op",
                                                "--keep", "Дата"])
            self.assertEqual(rc, 0)
            sc = pd.read_csv(Path(tmp) / "calls_scored.csv")
            self.assertEqual(len(sc), 75)
            self.assertTrue(sc["roles_swapped"].all())
            self.assertEqual(set(sc["topic_keywords"]), set(TOPICS))
            self.assertIn("Дата", sc.columns)
            self.assertIn(rude, sc["id"].head(3).tolist())
            self.assertIn("идиот", sc.set_index("id").loc[rude, "worst_operator_turn"])
            self.assertNotIn("auto-1", sc["id"].tolist())                         # автоответчик вместо разговора — пропущен
            self.assertNotIn("auto-2", sc["id"].tolist())                         # голосовой ассистент абонента
            self.assertNotIn("mono-1", sc["id"].tolist())                         # одна реплика — разговора нет
            load_script("run_calls").main([str(src), "--text", "text", "--id", "id", "--model", "tfidf", "--out", tmp,
                                           "--roles", "auto", "--topic", "Тема", "--min-topic", "10", "--center-on", "op",
                                           "--score", "sidak", "--min-words", "2"])
            sc = pd.read_csv(Path(tmp) / "calls_scored.csv")
            self.assertIn(rude, sc["id"].head(3).tolist())

    def test_run_calls_embed_resumes_from_chunks(self):
        mod = load_script("run_calls")
        words = ["вечером", "утром", "снова", "опять", "второй день", "после грозы", "у соседей тоже", "срочно"]
        texts = [f"{p} {w}" for ph in TOPICS.values() for p in ph for w in words] * 2      # 120 уникальных, каждая дважды
        with tempfile.TemporaryDirectory() as tmp:
            one, many, resumed = (Path(tmp) / x for x in ("one", "many", "resumed"))
            for d in (one, many, resumed):
                d.mkdir()
            ref = mod.embed(texts, "tfidf", one, "t", chunk=10_000)
            np.testing.assert_allclose(mod.embed(texts, "tfidf", many, "t", chunk=25), ref, atol=1e-6)
            self.assertEqual(list(many.glob("*.part*.npy")), [])           # куски убраны после сборки
            # прерванный прогон: первый кусок уже на диске — берётся оттуда, а не считается заново
            h = next(one.glob("emb_t_tfidf_*.npy")).stem.split("_")[-1]
            np.save(resumed / f"emb_t_tfidf_{h}.part000.npy", np.zeros((25, ref.shape[1]), dtype=np.float32))
            got = mod.embed(texts, "tfidf", resumed, "t", chunk=25)
            uniq = sorted(set(texts))
            first = {u for u in uniq[:25]}
            m = np.array([x in first for x in texts])
            self.assertTrue(np.all(got[m] == 0))
            np.testing.assert_allclose(got[~m], ref[~m], atol=1e-6)

    def test_run_calls_embed_reuses_previous_caches(self):
        """Второй прогон с частично новым набором реплик считает эмбеддинги только для новых текстов."""
        mod = load_script("run_calls")
        seen = []

        class Fake:
            def fit(self, texts):
                return self

            def encode(self, texts):
                seen.extend(texts)
                return np.array([[len(x), sum(map(ord, x)) % 97, 1.0] for x in texts], dtype=np.float32)

        mod.get_embedder, mod.ST_MODELS = (lambda model: Fake()), {"fake": None}
        a = [f"реплика {i}" for i in range(30)]
        b = a[10:] + [f"новая {i}" for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            ea = mod.embed(a, "fake", Path(tmp), "t")
            self.assertEqual(len(seen), 30)
            seen.clear()
            eb = mod.embed(b, "fake", Path(tmp), "t")
            self.assertEqual(sorted(seen), sorted(f"новая {i}" for i in range(5)))
            np.testing.assert_array_equal(eb[:20], ea[10:])
            np.testing.assert_array_equal(eb, Fake().encode(b))
            seen.clear()
            mod.embed(b, "fake", Path(tmp), "t")                             # тот же набор — прямо из кэша
            self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
