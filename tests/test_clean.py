"""Очистка: структура SMM, ПД, отсутствие ложных удалений, LLM-слой с заглушкой, кэш."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lurker.clean import LLMCleaner, RuleCleaner, TextCache, build_clean_messages, clean_texts, remove_boilerplate  # noqa: E402
from lurker.dirty import make_dirty  # noqa: E402
from lurker.llm import FakeLLM  # noqa: E402

SMM = ("https://vk.com/wall-38206512_92485?reply=1182\nПереписка:\nКлиент: как заказать ключи от домофона?\n"
       "Мы: Добрый день! Уже передали специалистам, вернёмся с ответом.\nКлиент: +7(925)4230148, перезвоните\n"
       "Мы: Заявка оформлена, номер 806462.\nИтог: спрашивает сколько стоит ключ от домофона, объяснили стоимость ключа\n"
       "Морозов Олег Николаевич, ул. Школьная, д. 102, кв. 134")
ONELINE = ("https://ok.ru/group/68450833/topic/432917516 | клиент: приставка ваша постоянно зависает / мы: напишите номер "
           "лицевого счёта / клиент: ул. Молодёжная, д. 64, кв. 179 | саммари: приставка зависает, перезагрузили приставку, помогло | "
           "ФИО: Олег Петрович Телефон: 89111251721 Адрес: ул. Молодёжная, д. 64, кв. 179 Договор: 5123300")
NO_SUMMARY = "личка ВК, скрин во вложении\nКлиент: не могу зайти в приложение, пишет неверный пароль\nМы: сбросили пароль\nАнна Смирнова, тел. +7 925 123-45-67"


class RuleCleanerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = RuleCleaner(ner=False)
        cls.ner = RuleCleaner(ner=True)

    def test_summary_extracted(self):
        for cleaner in (self.rules, self.ner):
            r = cleaner.clean(SMM)
            self.assertEqual(r.structure, "summary")
            self.assertEqual(r.text, "спрашивает сколько стоит ключ от домофона, объяснили стоимость ключа")
            r = cleaner.clean(ONELINE)
            self.assertEqual(r.text, "приставка зависает, перезагрузили приставку, помогло")

    def test_client_lines_when_no_summary(self):
        r = self.rules.clean(NO_SUMMARY)
        self.assertEqual(r.structure, "client_lines")
        self.assertIn("не могу зайти в приложение", r.text)
        self.assertNotIn("сбросили", r.text)
        self.assertNotIn("925", r.text)

    def test_pd_removed_from_plain_text(self):
        raw = ("Не работает домофон. Морозов Олег Николаевич, ул. Школьная, д. 102, кв. 134, тел. 89559512793, "
               "ivan1980@mail.ru, л/с 6765535, Иванова Е.С., Лесная 28-149, https://t.me/c/1/2. Передали в домофонию.")
        for cleaner in (self.rules, self.ner):
            r = cleaner.clean(raw)
            for bad in ("Морозов", "Школьная", "8955", "mail.ru", "6765535", "Иванова", "Лесная", "t.me", "28-149"):
                self.assertNotIn(bad, r.text, (cleaner.name, r.text))
            self.assertIn("Не работает домофон", r.text)
            self.assertIn("Передали в домофонию", r.text)
            self.assertGreaterEqual(r.removed["phone"], 1)
            self.assertGreaterEqual(r.removed["url"], 1)

    def test_ner_catches_free_names(self):
        r = self.ner.clean("нет интернета на Гагарина, когда починят?? Анна Смирнова, +7 925 123-45-67")
        self.assertNotIn("Анна", r.text)
        self.assertNotIn("Смирнова", r.text)
        self.assertIn("нет интернета", r.text)

    def test_no_false_removals(self):
        for s in ("периодически пропадает интернет, несколько раз в день", "wifi плохо ловит в дальней комнате, перезвонить",
                  "телевизор пишет нет сигнала, шумит, площадь покрытия", "Абонент обратился с проблемой: нет интернета",
                  "списали дважды абонплату, оформили возврат, 3-5 рабочих дней", "директор попросил перезвонить"):
            self.assertEqual(self.rules.clean(s).text, s)
            self.assertEqual(self.ner.clean(s).text, s)

    def test_placeholders_and_tidy(self):
        r = RuleCleaner(ner=False, placeholders=True).clean("звонил абонент, тел. 89559512793, https://vk.com/x , просил перезвонить")
        self.assertIn("[телефон]", r.text)
        self.assertIn("[ссылка]", r.text)
        self.assertNotIn("  ", r.text)
        self.assertEqual(self.rules.clean("Заявка №352901. Нет сигнала. Заявка №1.").text, "Нет сигнала. Заявка.")
        self.assertEqual(self.rules.clean("").text, "")
        self.assertEqual(self.rules.clean("-").text, "")


class FalseStructureTests(unittest.TestCase):
    """Случаи, найденные на открытых отзывах: служебные слова внутри обычного текста не должны включать разбор структуры."""

    def setUp(self):
        self.rc = RuleCleaner(ner=False)

    def test_itog_inside_review_is_not_summary(self):
        t = "С тех пор, как у меня появился автомобиль, я побывал в пяти сервисах. Итог: возвращался всегда к ним."
        self.assertEqual(self.rc.clean(t).text, t)

    def test_fio_label_takes_only_name(self):
        r = self.rc.clean("ФИО: Сидоров Иван Александрович Вчера обнаружила, что пропала карта, никто не смог объяснить.")
        self.assertEqual(r.text, "Вчера обнаружила, что пропала карта, никто не смог объяснить.")

    def test_client_label_at_end_is_not_dialog(self):
        r = self.rc.clean("Кредит 150000 рублей под 12.5% на 36 месяцев. Контактный центр не помог. Клиент: Елена Кузнецова")
        self.assertIn("Контактный центр не помог", r.text)
        self.assertIn("12.5%", r.text)
        self.assertNotIn("Кузнецова", r.text)

    def test_multiline_client_turn_kept_whole(self):
        t = ("https://vk.com/im?sel=1\nПереписка:\nМы: Здравствуйте!\nКлиент: Ходим в младшую группу.\n"
             "Площадки требуют ремонта.\nНо в целом довольны!\nМы: Спасибо за отзыв.\nИванова Мария Петровна, тел. 89001234567")
        self.assertEqual(self.rc.clean(t).text, "Ходим в младшую группу. Площадки требуют ремонта. Но в целом довольны!")

    def test_bare_digit_phones(self):
        self.assertEqual(self.rc.clean("Позвоните 378876232, жду. Платёж 5000 руб., ставка 12.5").text, "Позвоните, жду. Платёж 5000 руб., ставка 12.5")


    def test_email_with_dot_and_timestamps(self):
        out = RuleCleaner(ner=False).clean("пишите на ivan.petrov88@yandex.ru или elena@mail.ru. [14:02] Пользователь: не работает домофон").text
        self.assertNotIn("@", out)
        self.assertNotIn("ivan", out)
        self.assertNotIn("14:02", out)
        self.assertIn("не работает домофон", out)


class BoilerplateTests(unittest.TestCase):
    def test_removes_templates_keeps_content(self):
        topics = ["не работает домофон в первом подъезде", "пропал интернет после грозы вчера", "двойное списание абонплаты за март"]
        when = ["утром", "вечером", "ночью", "с понедельника", "после обеда", "в выходные", "постоянно"]
        texts = []
        for i in range(60):                     # содержание у каждого своё (тема × когда повторяется ≤ 3 раз)
            texts.append(f"Обращение №{1000 + i} от 0{i % 9 + 1}.03.2025\n[10:{i % 50 + 10}] Пользователь: {topics[i % 3]} {when[i % 7]}.\n"
                         f"[10:{i % 50 + 12}] Сотрудник поддержки (Анна): Благодарим, что написали нам. Обращение передано ответственному сотруднику.\n"
                         f"Статус: закрыто | Тег: #жалоба")
        out, removed = remove_boilerplate(texts)
        self.assertTrue(removed)
        for t, o in zip(texts, out):
            self.assertNotIn("Благодарим, что написали нам", o)
            self.assertNotIn("Статус: закрыто", o)
        self.assertIn("не работает домофон в первом подъезде утром", out[0])

    def test_free_text_untouched(self):
        texts = [f"Был тут {i} раз, всё понравилось, особенно {w}." for i, w in enumerate(["кофе", "персонал", "цены", "музей"] * 3)]
        out, removed = remove_boilerplate(texts)
        self.assertEqual(out, texts)
        self.assertEqual(removed, [])

    def test_alt_dirt_is_deterministic_and_marked(self):
        a = make_dirty(["Хороший сервис. Быстро починили."] * 20, "alt", seed=1, smm_share=0.5)
        b = make_dirty(["Хороший сервис. Быстро починили."] * 20, "alt", seed=1, smm_share=0.5)
        self.assertEqual([x.text for x in a], [x.text for x in b])
        self.assertEqual({x.kind for x in a}, {"alt", "crm"})
        self.assertTrue(all("Хороший сервис" in x.text for x in a))


class LLMCleanerTests(unittest.TestCase):
    def test_llm_layer_uses_prefiltered_text_and_json(self):
        seen = {}

        def reply(msgs):
            import json
            batch = json.loads(msgs[-1]["content"])
            seen.update(batch)
            return json.dumps({k: v.upper() for k, v in batch.items()}, ensure_ascii=False)
        cleaner = LLMCleaner(FakeLLM(reply), batch_size=2, prefilter=RuleCleaner(ner=False))
        out = cleaner.clean_many([SMM, "нет инета", "", "приставка зависает"])
        self.assertEqual(out[0].text, "СПРАШИВАЕТ СКОЛЬКО СТОИТ КЛЮЧ ОТ ДОМОФОНА, ОБЪЯСНИЛИ СТОИМОСТЬ КЛЮЧА")
        self.assertEqual(out[2].text, "")
        self.assertEqual(out[3].text, "ПРИСТАВКА ЗАВИСАЕТ")
        self.assertTrue(all("http" not in v and "Морозов" not in v for v in seen.values()))  # ПД не ушли в LLM
        self.assertEqual(cleaner.llm.calls, 2)

    def test_llm_failure_falls_back_to_rules(self):
        cleaner = LLMCleaner(FakeLLM("не json"), batch_size=10, prefilter=RuleCleaner(ner=False))
        out = cleaner.clean_many([SMM])
        self.assertEqual(out[0].text, "спрашивает сколько стоит ключ от домофона, объяснили стоимость ключа")
        self.assertEqual(cleaner.failed_batches, 1)

    def test_messages_have_examples(self):
        msgs = build_clean_messages(["a", "b"])
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertIn('"2": "b"', msgs[-1]["content"])


class CacheTests(unittest.TestCase):
    def test_cache_and_dedupe(self):
        class Counting(RuleCleaner):
            calls = 0

            def clean(self, text):
                Counting.calls += 1
                return super().clean(text)
        with tempfile.TemporaryDirectory() as d:
            cache = TextCache(Path(d) / "c.sqlite")
            c = Counting(ner=False)
            out = clean_texts(["нет интернета", "нет интернета", "", "тв не работает"], c, cache)
            self.assertEqual(out, ["нет интернета", "нет интернета", "", "тв не работает"])
            self.assertEqual(Counting.calls, 3)
            out2 = clean_texts(["тв не работает", "нет интернета", "новый"], Counting(ner=False), TextCache(Path(d) / "c.sqlite"))
            self.assertEqual(out2, ["тв не работает", "нет интернета", "новый"])
            self.assertEqual(Counting.calls, 4)


if __name__ == "__main__":
    unittest.main()
