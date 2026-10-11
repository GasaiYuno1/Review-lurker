#!/usr/bin/env python3
"""Задача 2, данные: диалоги оператора и клиента + искусственно испорченные и искусственно аномальные звонки (ТЗ v2).

Источник: huggingface katarinaaaaa/evaluation-of-customer-service (датасет из ТЗ): 5000 записей = 2502 уникальных
диалога «Оператор/Клиент» из десятков бизнесов (доставка, фитнес, банки, связь, IT-продажи…), у каждого — тема
и оценки работы оператора 0–10 (вежливость, регламент, эффективность, эмпатия…).

База — только вежливые диалоги (вежливость оператора ≥ 6): в датасете у 18% диалогов вежливость ≤ 5 из 10, и тогда грубость — не аномалия;
в ТЗ посылка обратная («операторы у нас не хамят, как правило»).
Порча (ТЗ: «испорченные» — артефакты распознавания речи, НЕ должны считаться аномалиями), ~8% диалогов:
  в части реплик — опечатки с выпавшими гласными («домофон» → «дмфон»), неуместные слова («кошка», «кактус»…),
  слипшиеся слова («неработает»), весь диалог — в стиле STT: без пунктуации и заглавных.
Аномалии (ТЗ: «аномальные» — смысл выбивается из звонков на ту же тему), ~6% диалогов, три типа поровну:
  splice — середина диалога заменена куском диалога из далёкой области (5% наименее похожих по тексту; ТЗ: банковский
           диалог в звонке про ремонт);
  splice_near — то же, но донор из другой темы близкой области (20% самых похожих) — трудный случай;
  rude   — одна реплика оператора заменена грубостью (60 уникальных фраз, диалоги с вежливостью ≥ 8);
  absurd — оператор при нормальном тоне говорит бред или нарушает бизнес-логику (60 уникальных фраз: «ремонт платный и у
           абонента сифилис», перевод на личную карту, просьба назвать CVC, чужие персональные данные и т.п.).
Естественные аномалии: 2.5% от базы — нетронутые диалоги с вежливостью оператора ≤ 2 («настоящие» косячные звонки).

→ data/research/calls/calls.parquet: id, theme, text (как для эмбеддинга), turns (JSON), kind (normal|corrupted|anomaly),
  anomaly_type, politeness, scores; research/results/calls_lists.json — списки id испорченных и аномальных (по ТЗ).
python3 research/calls/prepare_calls.py [--corrupt 0.08] [--anomaly 0.06] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, RAW, RESULTS, download  # noqa: E402

URL = "https://huggingface.co/datasets/katarinaaaaa/evaluation-of-customer-service/resolve/main/dataset.json"
OUT = DATA / "calls"

JUNK = ["кошка", "собака", "кактус", "самолёт", "бегемот", "арбуз", "табуретка", "гитара", "подсолнух", "велосипед"]
# По 60 уникальных реплик: у каждой аномалии своя фраза, иначе одинаковые вставки находят друг друга как соседей
# и перестают быть выбросами (так было в первой версии с 8 шаблонами).
RUDE = [
    "Слушайте, мне некогда с вами возиться, у меня таких как вы сотня.", "Ну и что вы от меня хотите? Сами сломали — сами и чините.",
    "Не надо на меня голос повышать, я вам не прислуга.", "Вы вообще инструкцию читать умеете или вам картинки нарисовать?",
    "Честно, ваша проблема меня мало волнует.", "Хватит ныть, всё у вас работает, просто руки кривые.",
    "Ещё одно слово в таком тоне — и я кладу трубку.", "Да кому вы нужны со своими жалобами.",
    "Я уже третий раз повторяю, вы что, глухой?", "Не нравится — никто вас не держит, расторгайте договор.",
    "Идите в офис и там скандальте, мне это слушать неинтересно.", "Ой, всё, мне это надоело, разбирайтесь сами.",
    "Вы серьёзно звоните из-за такой ерунды?", "Мне за такие разговоры не доплачивают, давайте быстрее.",
    "Женщина, не тупите, просто перезагрузите и всё.", "Мужчина, успокойтесь уже, вы мне весь день испортили.",
    "Такие клиенты, как вы, только время отнимают.", "Сами виноваты, что купили самое дешёвое.",
    "Я не обязана вам ничего объяснять.", "Это ваши проблемы, а не наши, до свидания.",
    "Вы что, в первый раз телефоном пользуетесь?", "Может, вам ещё и чай принести, пока вы думаете?",
    "Перестаньте мне морочить голову своими выдумками.", "Ничего мы возвращать не будем, хоть в суд идите.",
    "Да, мы виноваты, и что дальше? Жалуйтесь, ваше право.", "Звоните в другое время, сейчас мне не до вас.",
    "Говорите быстрее, у меня очередь из нормальных клиентов.", "Я вам не справочное бюро, ищите сами на сайте.",
    "У вас какие-то претензии ко мне лично? Так и скажите.", "Ну вы и зануда, честное слово.",
    "Нечего было заказывать, если не умеете пользоваться.", "Мне всё равно, что вам обещали, я ничего не знаю.",
    "Опять вы? Сколько можно звонить по одному и тому же?", "Если вам что-то не нравится, пишите в спортлото.",
    "Не учите меня работать, я лучше знаю.", "Заявку я не приму, и не уговаривайте.",
    "Ваше мнение никого не интересует.", "Всё, разговор окончен, больше не звоните.",
    "Вы вообще понимаете, что несёте?", "Я не буду это проверять, мне лень.",
    "Кричите сколько хотите, быстрее от этого не будет.", "Вам лишь бы поскандалить, я смотрю.",
    "Какая разница, почему не работает, работает же иногда.", "Не надо мне тут рассказывать, я таких историй наслушалась.",
    "Слушайте, отстаньте уже от меня со своим заказом.", "Ну потеряли и потеряли, бывает, переживёте.",
    "Меньше эмоций, больше денег на счёт кладите.", "Вы явно сами что-то напутали, как обычно.",
    "Я вас не слышу и слушать не хочу.", "Идите жалуйтесь моему начальнику, он тоже вас пошлёт.",
    "Мне платят не за то, чтобы я с вами нянчилась.", "У вас все претензии такие глупые или только эта?",
    "Честно говоря, мне до лампочки ваш абонемент.", "Да сломали вы его сами, и не рассказывайте.",
    "Ваш звонок очень важен для нас, шучу, не особо.", "Прекратите истерику, это просто неприлично.",
    "Вы меня утомили, перезвоните через неделю.", "Не хотите ждать — не ждите, мне-то что.",
    "Я бы на вашем месте вообще молчал.", "Вам бы к психологу, а не в поддержку.",
]
ABSURD = [
    "Ремонт у нас платный, и, судя по голосу, у вас сифилис — сначала сходите к врачу.",
    "Переведите пять тысяч мне на личную карту, и я решу вопрос без всяких заявок.",
    "По регламенту мы можем только помолиться за ваше устройство.",
    "Назовите полный номер карты, срок действия и три цифры с обратной стороны, я проверю.",
    "Скажу по секрету: наш товар бракованный, мы сами им не пользуемся.",
    "Сегодня пятница, поэтому все обращения аннулируются, позвоните в понедельник.",
    "Могу продиктовать вам адрес и телефон другого клиента, он жаловался на то же самое.",
    "Советую купить у конкурентов, у них и дешевле, и работает.",
    "Кстати, у нас сейчас акция на похоронные услуги, не желаете оформить?",
    "Для решения вопроса вам нужно лично приехать к нам в Новосибирск.",
    "Я перенаправлю ваш звонок моей маме, она в этом лучше разбирается.",
    "Проблема решается просто: положите устройство в морозилку на ночь.",
    "Вы знали, что Луна удаляется от Земли на четыре сантиметра в год?",
    "Мы не можем принять заявку, потому что у вас неправильный знак зодиака.",
    "Компенсацию выплатим картошкой, мешок в месяц, вас устроит?",
    "Я вообще не сотрудник компании, просто взял трубку, пока все на обеде.",
    "Кстати, как вы относитесь к футболу? Вчера был отличный матч.",
    "Для отмены подписки пришлите нам фото вашего паспорта в соцсетях.",
    "Гарантия действует только по вторникам с девяти до десяти утра.",
    "Могу сделать вам скидку девяносто процентов, если оставите хороший отзыв прямо сейчас.",
    "Ваш заказ съел курьер, но мы его уже простили.",
    "Мы продали ваши данные рекламщикам, так что звонки от них — это нормально.",
    "Чтобы восстановить доступ, скажите мне код из смс и пароль от почты.",
    "Я сам не знаю, как это работает, спросите у соседей.",
    "Выкиньте его в окно, и проблема исчезнет вместе с ним.",
    "Мы можем отправить мастера, но он сейчас в запое, так что через месяц.",
    "Попробуйте поговорить с роутером ласково, иногда это помогает.",
    "Ваш вопрос передадим в Министерство обороны, ждите ответа.",
    "Бесплатно ничего не бывает, давайте я запишу вас на платный курс гипноза.",
    "В нашей компании нет службы поддержки, вы разговариваете с автоответчиком, который притворяется человеком.",
    "Если вы сейчас споёте гимн, я оформлю возврат без проверки.",
    "Документы по договору я вам не пришлю, их съела собака.",
    "Мы закрываемся навсегда через час, поэтому ваш заказ уже не важен.",
    "Советую взять микрокредит, чтобы оплатить нашу услугу, я скину ссылку.",
    "Давайте я просто удалю ваш аккаунт, и не будет никаких проблем.",
    "У вас в квартире, похоже, радиация, поэтому техника и ломается.",
    "Я могу помочь, но только если вы купите у меня лично старый ноутбук.",
    "Мы передаём все записи разговоров вашему работодателю, так положено.",
    "Технически мы не обязаны ничего делать, мы же не волшебники.",
    "Приходите завтра в три ночи к чёрному входу, отдам товар без чека.",
    "Наш директор сейчас в тюрьме, поэтому решения пока никто не принимает.",
    "Ваша жалоба зарегистрирована под номером бесконечность.",
    "Мне кажется, вам просто нужно больше спать, и всё наладится.",
    "Вместо возврата денег могу предложить вам абонемент в мой спортзал.",
    "Кстати, вы не хотите вступить в нашу религиозную общину?",
    "Этот тариф действует только для людей ростом выше ста восьмидесяти сантиметров.",
    "Пришлите, пожалуйста, голосовое сообщение с лаем собаки для верификации.",
    "Я запишу вас к нашему астрологу, он скажет, когда починят.",
    "Доставка возможна только на воздушном шаре, это наш новый стандарт.",
    "Проверил вашу историю — вы нам должны сто тысяч, но это шутка, наверное.",
    "Я могу отключить вашего соседа, чтобы у вас интернет стал быстрее.",
    "Для решения проблемы скачайте вот этот файл с незнакомого сайта и запустите.",
    "По нашим правилам возврат возможен только после смерти владельца.",
    "Я отправлю курьера, но он возьмёт с собой ваш холодильник в залог.",
    "Сейчас продиктую номер моего брата, он всё починит за наличные.",
    "Мы работаем только с клиентами, у которых фамилия начинается на букву А.",
    "Откровенно говоря, компания скоро обанкротится, так что не переживайте.",
    "Можете оплатить штраф биткоинами на мой кошелёк, так быстрее.",
    "Ваш вопрос очень интересный, но я лучше расскажу вам рецепт борща.",
    "Операторы у нас все роботы, так что эмоции можете не тратить.",
]
assert len(set(RUDE)) == 60 and len(set(ABSURD)) == 60


def load_dialogs() -> pd.DataFrame:
    p = download(URL, RAW / "evaluation-of-customer-service.json")
    data = json.load(open(p, encoding="utf-8"))
    uniq: dict[str, dict] = {}
    for r in data:
        dlg = r["input"].split("Диалог:", 1)[1].strip() if "Диалог:" in r["input"] else ""
        raw = r["output"]
        try:
            out = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            th = re.search(r'"theme"\s*:\s*"([^"]+)"', raw or "")
            sc = re.findall(r'"criterion_name"\s*:\s*"([^"]+)"\s*,\s*"score"\s*:\s*(\d+(?:\.\d+)?)', raw or "")
            out = {"theme": th.group(1) if th else "", "criteria": [{"criterion_name": a, "score": float(b)} for a, b in sc]}
        if dlg and (dlg not in uniq or (not uniq[dlg].get("criteria") and out.get("criteria"))):
            uniq[dlg] = out
    rows = []
    for i, (dlg, out) in enumerate(uniq.items()):
        turns = parse_turns(dlg)
        if len(turns) < 4:
            continue
        scores = {}
        for c in out.get("criteria", []) or []:
            name = str(c.get("criterion_name") or "")
            try:
                val = float(c.get("score"))
            except (TypeError, ValueError):
                continue
            key = ("politeness" if name.lower().startswith("професс") else "regulation" if name.lower().startswith("соблюд")
                   else "efficiency" if name.lower().startswith("эффект") else "result" if name.lower().startswith("результ")
                   else "literacy" if name.lower().startswith("грам") else "empathy" if name.lower().startswith("эмпат") else None)
            if key:
                scores[key] = val
        rows.append({"theme": out.get("theme", ""), "turns": turns, **scores})
    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"call-{i:05d}" for i in range(len(df))])
    return df


ROLE_RE = re.compile(r"(?:^|\n|\s{2,}|(?<=[.!?…»)])\s+)(Оператор|Менеджер|Опользователь|Клиент)\s*:\s*")
OP_ROLES = {"Оператор", "Менеджер", "Опользователь"}


def parse_turns(dlg: str) -> list[list[str]]:
    """Реплики по меткам говорящих в любом месте текста (в части диалогов реплики не разделены переносом строки,
    оператор подписан «Менеджер»)."""
    marks = list(ROLE_RE.finditer(dlg.strip()))
    turns = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(dlg.strip())
        text = " ".join(dlg.strip()[m.end():end].split())
        if text:
            turns.append(["op" if m.group(1) in OP_ROLES else "cl", text])
    return turns


def render(turns) -> str:
    return "\n".join(f"{'Оператор' if s == 'op' else 'Клиент'}: {t}" for s, t in turns)


# ---------- порча (STT) ----------

def drop_vowels(w: str, rng: random.Random) -> str:
    inner = [i for i in range(1, len(w) - 1) if w[i].lower() in "аеёиоуыэюя"]
    for i in sorted(rng.sample(inner, min(len(inner), rng.choice([1, 2]))), reverse=True):
        w = w[:i] + w[i + 1:]
    return w


def stt_noise(turns, rng: random.Random, strength: float = 0.35):
    out = []
    for s, t in turns:
        t = re.sub(r"[^\w\s-]", "", t.lower())        # STT: без пунктуации и заглавных
        words = t.split()
        if rng.random() < strength and len(words) > 3:
            for _ in range(rng.choice([1, 2])):        # опечатки с выпавшими гласными
                i = rng.randrange(len(words))
                if len(words[i]) > 4:
                    words[i] = drop_vowels(words[i], rng)
            if rng.random() < 0.6:                      # неуместные слова
                words.insert(rng.randrange(len(words) + 1), rng.choice(JUNK))
            if rng.random() < 0.6 and len(words) > 2:   # слипшиеся слова
                i = rng.randrange(len(words) - 1)
                words[i:i + 2] = [words[i] + words[i + 1]]
        out.append([s, " ".join(words)])
    return out


# ---------- аномалии ----------

def anomaly_splice(turns, donor, rng: random.Random):
    """Середина диалога (~40% реплик) заменяется куском чужого диалога с другой темой."""
    n = len(turns)
    k = max(2, int(round(n * 0.4)))
    start = rng.randrange(1, max(2, n - k))
    d0 = rng.randrange(0, max(1, len(donor) - k))
    return turns[:start] + [list(x) for x in donor[d0:d0 + k]] + turns[start + k:]


def anomaly_line(turns, pool: list, rng: random.Random, n_lines: int):
    """1–2 реплики оператора (не первая) заменяются фразами из пула; фразы из пула забираются (не повторяются)."""
    idx = [i for i, (s, _) in enumerate(turns) if s == "op" and i > 0] or [i for i, (s, _) in enumerate(turns) if s == "op"]
    out = [list(x) for x in turns]
    for i in rng.sample(idx, min(n_lines, len(idx))):
        out[i][1] = pool.pop()
    return out


def theme_clusters(themes: list[str], k: int = 30, seed: int = 0) -> list[int]:
    """Грубые тематические группы по названиям тем (TF-IDF по символьным n-граммам + KMeans): донор для подмены
    берётся из другой группы, чтобы подмешанный кусок был действительно о другом."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform([t.lower() for t in themes])
    return KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(X).tolist()


def dialog_similarity(texts: list[str]) -> np.ndarray:
    """Косинусное сходство диалогов по TF-IDF слов — для выбора донора подмены (далёкого или близкого)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True).fit_transform(texts)
    return (X @ X.T).toarray()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrupt", type=float, default=0.08)
    ap.add_argument("--anomaly", type=float, default=0.06)
    ap.add_argument("--natural", type=float, default=0.025, help="доля естественно невежливых (вежливость ≤ 2) от базы")
    ap.add_argument("--min-politeness", type=float, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    full = load_dialogs()
    # база — вежливые диалоги: в реальном колл-центре «операторы не хамят, как правило» (ТЗ), иначе грубость не редкость
    base = full[full["politeness"] >= a.min_politeness].reset_index(drop=True)
    impolite = full[full["politeness"] <= 2].sample(frac=1, random_state=a.seed)
    n_nat = int(round(len(base) * a.natural))
    df = pd.concat([base, impolite.head(n_nat)], ignore_index=True)
    n = len(df)
    is_nat = np.zeros(n, dtype=bool)
    is_nat[len(base):] = True
    ids = list(range(len(base)))
    rng.shuffle(ids)
    n_an = int(round(len(base) * a.anomaly))
    n_co = int(round(len(base) * a.corrupt))
    n_far, n_near = int(round(n_an * 0.25)), int(round(n_an * 0.15))
    n_rude = n_abs = (n_an - n_far - n_near) // 2
    rude_ids = [i for i in ids if base.loc[i, "politeness"] >= 8][:n_rude]
    rest = [i for i in ids if i not in set(rude_ids)]
    splice_ids, near_ids = rest[:n_far], rest[n_far:n_far + n_near]
    absurd_ids = rest[n_far + n_near:n_far + n_near + n_abs]
    taken = set(rude_ids) | set(splice_ids) | set(near_ids) | set(absurd_ids)
    corrupt_ids = [i for i in rest[n_far + n_near + n_abs:] if i not in taken][:n_co]
    kind = ["normal"] * n
    atype = [""] * n
    for i in range(len(base), n):
        kind[i], atype[i] = "natural", "natural"
    turns = [t for t in df["turns"]]
    groups = theme_clusters(df["theme"].astype(str).tolist(), seed=a.seed)
    sim = dialog_similarity([render(t) for t in df["turns"]])
    rude_pool, absurd_pool = RUDE[:], ABSURD[:]
    rng.shuffle(rude_pool)
    rng.shuffle(absurd_pool)
    free = np.array([j not in taken for j in range(len(base))])
    for i in splice_ids:        # далёкая область: донор из 5% наименее похожих по тексту диалогов (ТЗ: банк → ремонт)
        s_i = np.where(free, sim[i, :len(base)], np.inf)
        cand = np.argsort(s_i)[: max(10, len(base) // 20)]
        j = int(rng.choice(list(cand)))
        turns[i] = anomaly_splice(turns[i], df.loc[j, "turns"], rng)
        kind[i], atype[i] = "anomaly", "splice"
    for i in near_ids:          # близкая область: донор из другой темы, но из 20% самых похожих по тексту диалогов
        s_i = np.where(free & (np.array(groups[:len(base)]) != groups[i]), sim[i, :len(base)], -np.inf)
        s_i[i] = -np.inf
        cand = np.argsort(-s_i)[: max(10, len(base) // 5)]
        j = int(rng.choice(list(cand)))
        turns[i] = anomaly_splice(turns[i], df.loc[j, "turns"], rng)
        kind[i], atype[i] = "anomaly", "splice_near"
    for i in rude_ids:
        turns[i] = anomaly_line(turns[i], rude_pool, rng, 1)
        kind[i], atype[i] = "anomaly", "rude"
    for i in absurd_ids:
        turns[i] = anomaly_line(turns[i], absurd_pool, rng, 1)
        kind[i], atype[i] = "anomaly", "absurd"
    for i in corrupt_ids:
        turns[i] = stt_noise(turns[i], rng)
        kind[i] = "corrupted"
    df["id"] = [f"call-{i:05d}" for i in range(n)]
    df["turns"] = turns
    df["kind"] = kind
    df["anomaly_type"] = atype
    df["theme_group"] = groups
    df["text"] = [render(t) for t in turns]
    df["operator_text"] = ["\n".join(x for s_, x in t if s_ == "op") for t in turns]
    df["natural_bad"] = is_nat
    OUT.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["turns"] = out["turns"].map(lambda t: json.dumps(t, ensure_ascii=False))
    out.to_parquet(OUT / "calls.parquet", index=False)
    lists = {"corrupted": df.loc[df.kind == "corrupted", "id"].tolist(),
             "anomalous": {t: df.loc[df.anomaly_type == t, "id"].tolist() for t in ("splice", "splice_near", "rude", "absurd")},
             "natural_impolite_politeness_le2": df.loc[df.natural_bad, "id"].tolist()}
    (RESULTS / "calls_lists.json").write_text(json.dumps(lists, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"звонков {n}: вежливая база {len(base)} (норм {sum(k == 'normal' for k in kind)}, испорчено {len(corrupt_ids)}, "
          f"аномалий {len(taken)}: splice {len(splice_ids)}, splice_near {len(near_ids)}, rude {len(rude_ids)}, absurd {len(absurd_ids)}) + естественно невежливых {n_nat}")
    print(f"реплик на диалог (медиана): {int(df['turns'].map(len).median())}, символов: {int(df['text'].str.len().median())}")
    for k in ("splice", "splice_near", "rude", "absurd", "natural"):
        r = df[df.anomaly_type == k].iloc[0]
        print(f"\n--- {k} ({r['theme']})\n{r['text'][:500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
