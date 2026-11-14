"""Что пришло из прогона на прод-данных: всплески тем по дням и их слова, сценарий аварии, словарь брани, проверка
звонков LLM по чек-листу (с FakeLLM, без сети) и эти шаги в research/run_calls.py."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from lurker import llm as llm_mod  # noqa: E402
from lurker.anomaly import rude_hits  # noqa: E402
from lurker.checklist import check_call, format_dialog, note_kind, parse_answer  # noqa: E402
from lurker.config import LLMConfig  # noqa: E402
from lurker.incidents import daily_counts, daily_spikes, outage_days, spike_words  # noqa: E402
from lurker.llm import FakeLLM, LLMClient  # noqa: E402
from test_run_scripts import TOPICS, load_script  # noqa: E402


class IncidentsTests(unittest.TestCase):
    def test_spike_found_with_words_and_outage(self):
        rnd = random.Random(0)
        days, topics, texts = [], [], []
        for d in range(1, 15):
            day = f"2026-09-{d:02d}"
            for t in ("интернет", "домофон", "оплата", "тв", "ключи", "договор"):
                n = 60 if (t == "интернет" and d == 7) else 10
                for _ in range(n):
                    days.append(day); topics.append(t)
                    texts.append("ведутся аварийные работы приносим извинения за неудобства кабель повредили"
                                 if (t == "интернет" and d == 7) else f"{t} вопрос {rnd.choice(['счёт', 'роутер', 'ключ'])}")
        sp = daily_spikes(daily_counts(days, topics))      # тем 6: у темы с большой долей всплеск в 2 раза невозможен
        self.assertEqual(sp[["day", "topic"]].values.tolist(), [["2026-09-07", "интернет"]])
        self.assertGreater(sp.ratio.iloc[0], 2)
        w = spike_words(texts, days, topics, sp)
        self.assertIn("ведутся", w.words.iloc[0])
        od = outage_days(texts, days)
        self.assertEqual(od.share.idxmax(), "2026-09-07")
        self.assertEqual(int(od.outage.sum()), 60)

    def test_no_spikes_on_flat_counts(self):
        days = [f"2026-09-{d:02d}" for d in range(1, 11) for _ in range(30)]
        topics = ["а", "б", "в"] * 100
        self.assertEqual(len(daily_spikes(daily_counts(days, topics))), 0)


class RudeTests(unittest.TestCase):
    def test_rude_hits_by_role(self):
        t = [("op", "здравствуйте слушаю вас"), ("cl", "да вы задолбали уже со своим интернетом"),
             ("op", "заткнись и слушай меня"), ("op", "идиот")]        # последняя короче 3 слов — мимо
        h = rude_hits(t)
        self.assertEqual(h["op"], [(2, "заткнись")])
        self.assertEqual(h["cl"], [(1, "задолбали")])


class ChecklistTests(unittest.TestCase):
    def test_parse_answer(self):
        self.assertEqual(parse_answer('{"label": "a", "outcome": "p", "repeat": "1", "note": "хамит"}'),
                         {"label": "A", "outcome": "P", "repeat": 1, "note": "хамит"})
        self.assertEqual(parse_answer("```json\n{\"label\": \"N\", \"outcome\": \"D\", \"repeat\": 0}\n```")["label"], "N")
        self.assertEqual(parse_answer('{"label": "Z", "outcome": "?"}'), {"label": "?", "outcome": "U", "repeat": 0, "note": ""})
        self.assertEqual(parse_answer("метка S, потому что конфликт")["label"], "S")

    def test_check_call_prompt_and_answer(self):
        seen = []

        def reply(msgs):
            seen.append(msgs[-1]["content"])
            return json.dumps({"label": "A", "outcome": "X", "repeat": 0, "note": "отказ помогать"})
        r = check_call([("op", "здравствуйте"), ("cl", "не работает интернет"), ("op", "это не моя проблема")],
                       FakeLLM(reply), topic="Интернет")
        self.assertEqual(r["label"], "A")
        self.assertIn("ОП: это не моя проблема", seen[0])
        self.assertIn("Тема из разметки: Интернет", seen[0])
        self.assertIn("КЛ: не работает интернет", format_dialog([("cl", "не   работает интернет")]))

    def test_note_kind(self):
        self.assertEqual(note_kind("оператор груб и перебивает клиента"), "грубость")
        self.assertEqual(note_kind("неверно назвал стоимость выезда"), "неверная информация")
        self.assertEqual(note_kind("отказ оформить заявку"), "отказ помогать")
        self.assertEqual(note_kind("ушёл в личный разговор"), "личное")
        self.assertEqual(note_kind(""), "прочее")

    def test_temperature_null_not_sent(self):
        bodies = []
        for temp, expect in ((None, False), (0.0, True)):
            c = LLMClient(LLMConfig(temperature=temp, json_mode=False))
            with mock.patch.object(c, "_request", side_effect=lambda body: bodies.append(body) or "ok"):
                c.chat([{"role": "user", "content": "привет"}])
            self.assertEqual("temperature" in bodies[-1], expect)


class FakeClient(FakeLLM):
    def __init__(self, cfg=None, cache_dir=None):
        super().__init__(lambda msgs: json.dumps({"label": "A" if "идиот" in msgs[-1]["content"] else "N",
                                                  "outcome": "D", "repeat": 0, "note": "оскорбление" if "идиот" in msgs[-1]["content"] else ""}))
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}


class RunCallsProdOptionsTests(unittest.TestCase):
    def test_date_spikes_rude_and_llm_checklist(self):
        rnd = random.Random(3)
        answers = {"домофон": "оформил заявку на ремонт домофона", "интернет": "перезагрузите роутер и проверьте кабель",
                   "оплата": "сделал перерасчёт вернём на счёт"}
        rows = []
        names = {"домофон": ["домофон", "ключи"], "интернет": ["интернет", "тв"], "оплата": ["оплата", "договор"]}
        for (topic, phrases), name in [(tp, n) for tp in TOPICS.items() for n in names[tp[0]]]:
            for i in range(40):
                day = f"2026-09-{1 + i % 8:02d}"
                op = answers[topic]
                if name == "интернет" and i >= 20:                  # 7-й день: авария, интернет-звонков в разы больше
                    day, op = "2026-09-07", "ведутся аварийные работы приносим извинения за неудобства"
                rows.append({"id": f"{name}-{i}", "Тема": name, "Дата": day,
                             "text": f"Клиент: {rnd.choice(phrases)}\nОператор: здравствуйте слушаю вас\nОператор: {op}"})
        rows[5]["text"] += "\nОператор: сами виноваты идиот отстаньте от меня"
        with tempfile.TemporaryDirectory() as tmp:
            src, cfg = Path(tmp) / "c.csv", Path(tmp) / "llm.yaml"
            pd.DataFrame(rows).to_csv(src, index=False)
            cfg.write_text("llm:\n  base_url: http://localhost:1\n  model: fake\n", encoding="utf-8")
            with mock.patch.object(llm_mod, "LLMClient", FakeClient):
                rc = load_script("run_calls").main([str(src), "--text", "text", "--id", "id", "--model", "tfidf", "--out", tmp,
                                                    "--topic", "Тема", "--min-topic", "10", "--date", "Дата",
                                                    "--llm", str(cfg), "--llm-workers", "2"])
            self.assertEqual(rc, 0)
            sp = pd.read_csv(Path(tmp) / "spikes.csv")
            self.assertIn(("2026-09-07", "интернет"), list(zip(sp.day, sp.topic)))
            od = pd.read_csv(Path(tmp) / "outage_days.csv", index_col=0)
            self.assertEqual(od.share.idxmax(), "2026-09-07")
            sc = pd.read_csv(Path(tmp) / "calls_scored.csv").set_index("id")
            self.assertEqual(sc.loc["домофон-5", "rude_op"], 1)
            self.assertEqual(sc.loc["домофон-5", "llm_label"], "A")
            self.assertEqual(sc.loc["домофон-5", "llm_kind"], "грубость")
            self.assertEqual((sc.llm_label == "A").sum(), 1)
            self.assertTrue((Path(tmp) / "checklist.csv").exists())
            llm_md = (Path(tmp) / "top.md").read_text(encoding="utf-8").split("# Нарушения по чек-листу LLM")[1]
            self.assertIn("Грубость — 1", llm_md)
            self.assertIn("домофон-5", llm_md)


if __name__ == "__main__":
    unittest.main()
