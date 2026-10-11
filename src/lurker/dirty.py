"""«Замусоривание» текстов для экспериментов (ТЗ v2, задача 1): в чистые отзывы подмешивается то, что встречается
в реальных тикетах, чтобы проверить, насколько мусор портит эмбеддинги и кластеры и насколько помогает очистка.

Уровни:
  tz   — ровно просьба ТЗ: в каждый текст ссылка, имя и номер телефона (случайные 9–11 цифр или +7-формат),
         в случайном порядке и в случайном месте (начало / середина / конец);
  smm  — формат SMM-тикета: «ссылка → переписка → (саммари) → ПД»: текст отзыва становится репликой клиента,
         вокруг — шаблонные реплики оператора (одни и те же фразы у сотен текстов — главный источник ложной близости),
         блок персональных данных (ФИО, телефон, адрес, email, лицевой счёт);
  mix  — случайно tz или smm (доля smm = smm_share);
  alt  — проверка обобщения очистки: тот же смысл (ПД, ссылки, служебная обвязка), но в других форматах, под которые
         правила очистки не писались: ссылки без https, телефоны со скобками и пробелами, «Фамилия И.О.», ники, e-mail
         в тексте, номера обращений и даты; доля smm_share — выгрузка из CRM («Обращение №…», «[14:02] Пользователь:»,
         «Сотрудник поддержки (Анна):», «--- Резюме ---», «Контактные данные:», «Статус: закрыто»).
Всё детерминировано по seed. Возвращается и текст, и список того, что подмешано (для контроля очистки).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

FIRST_M = ["Александр", "Дмитрий", "Сергей", "Андрей", "Алексей", "Иван", "Максим", "Евгений", "Николай", "Олег", "Павел", "Роман"]
FIRST_F = ["Елена", "Ольга", "Наталья", "Татьяна", "Анна", "Мария", "Ирина", "Светлана", "Юлия", "Екатерина", "Дарья", "Ксения"]
PATR_M = ["Александрович", "Сергеевич", "Владимирович", "Николаевич", "Игоревич", "Петрович", "Андреевич"]
PATR_F = ["Александровна", "Сергеевна", "Владимировна", "Николаевна", "Игоревна", "Петровна", "Андреевна"]
SURNAMES = ["Иванов", "Петров", "Сидоров", "Кузнецов", "Смирнов", "Попов", "Лебедев", "Козлов", "Новиков", "Морозов",
            "Волков", "Соловьёв", "Зайцев", "Павлов", "Семёнов", "Голубев", "Виноградов", "Богданов", "Воробьёв", "Фёдоров"]
STREETS = ["Ленина", "Гагарина", "Мира", "Советская", "Кирова", "Пушкина", "Строителей", "Молодёжная", "Садовая", "Лесная"]
GREETINGS = ["Здравствуйте! Подскажите, пожалуйста, номер договора, проверим.",
             "Добрый день! Приносим извинения за неудобства. Уточните, пожалуйста, данные в личных сообщениях.",
             "Здравствуйте! Спасибо за обращение, передали информацию специалистам.",
             "Добрый день! Напишите, пожалуйста, ваш номер телефона, мы свяжемся с вами."]
REPLIES = ["Спасибо, передали в профильный отдел, специалист свяжется с вами в течение дня.",
           "Заявка зарегистрирована, ожидайте звонка специалиста.",
           "Благодарим за обратную связь! Информацию передали руководителю.",
           "Проверили информацию, ответ направили в личные сообщения."]
CHANNELS = ["личка ВК", "комментарий под постом", "директ", "отзыв на картах", "чат поддержки"]


def fio(rng: random.Random) -> str:
    male = rng.random() < 0.5
    sur = rng.choice(SURNAMES) + ("" if male else "а")
    first = rng.choice(FIRST_M if male else FIRST_F)
    patr = rng.choice(PATR_M if male else PATR_F)
    return rng.choice([f"{sur} {first} {patr}", f"{first} {sur}", f"{sur} {first[0]}.{patr[0]}.", f"{first} {patr}", first])


def phone(rng: random.Random) -> str:
    """Номер: как просит ТЗ — «рандомные 9-10-11 цифр», плюс типичные российские форматы."""
    kind = rng.random()
    if kind < 0.5:
        n = rng.choice([9, 10, 11])
        return "".join(rng.choice("0123456789") for _ in range(n))
    d = f"9{rng.randrange(10, 99)}{rng.randrange(1000000, 9999999)}"
    return rng.choice([f"+7 {d[:3]} {d[3:6]}-{d[6:8]}-{d[8:]}", f"8{d}", f"+7({d[:3]}){d[3:]}", f"+7{d}"])


def link(rng: random.Random) -> str:
    g, p, i = rng.randrange(10 ** 7, 10 ** 8), rng.randrange(1000, 99999), rng.randrange(10 ** 8, 10 ** 9)
    return rng.choice([f"https://vk.com/wall-{g}_{p}?reply={rng.randrange(100, 9999)}", f"https://vk.com/im?sel={i}",
                       f"https://t.me/c/{i}/{rng.randrange(100, 99999)}", f"https://ok.ru/group/{g}/topic/{i}",
                       f"https://2gis.ru/city/firm/{i}/tab/reviews", f"https://yandex.ru/maps/org/{i}/reviews/",
                       f"https://www.banki.ru/services/responses/bank/response/{i}/", f"https://otzovik.com/review_{p}.html"])


def address(rng: random.Random) -> str:
    return rng.choice([f"ул. {rng.choice(STREETS)}, д. {rng.randrange(1, 120)}, кв. {rng.randrange(1, 300)}",
                       f"{rng.choice(STREETS)} {rng.randrange(1, 120)}-{rng.randrange(1, 300)}"])


def email(rng: random.Random) -> str:
    return f"{rng.choice(['ivan', 'olga', 'sergey', 'anna', 'dmitry', 'elena'])}{rng.randrange(1970, 2005)}@{rng.choice(['mail.ru', 'yandex.ru', 'gmail.com'])}"


@dataclass
class Dirty:
    text: str
    kind: str                       # tz | smm
    added: dict = field(default_factory=dict)   # что подмешано: link, name, phone, ...


def dirty_tz(text: str, rng: random.Random) -> Dirty:
    """Ссылка, имя, телефон — каждый в случайное место: начало, конец или между предложениями."""
    parts = {"link": link(rng), "name": fio(rng), "phone": phone(rng)}
    labels = {"link": "", "name": rng.choice(["", "Клиент: ", "ФИО: ", "Автор: "]), "phone": rng.choice(["", "тел. ", "Телефон: ", "контакт "])}
    sentences = [s for s in _split_sentences(text) if s]
    for key in rng.sample(list(parts), 3):
        piece = labels[key] + parts[key]
        pos = rng.choice(["start", "end", "middle"])
        if pos == "start" or not sentences:
            sentences.insert(0, piece)
        elif pos == "end":
            sentences.append(piece)
        else:
            sentences.insert(rng.randrange(1, len(sentences) + 1), piece)
    sep = rng.choice([" ", " ", "\n", " | "])
    return Dirty(sep.join(sentences), "tz", parts)


def dirty_smm(text: str, rng: random.Random, summary: Optional[str] = None) -> Dirty:
    """SMM-формат: ссылка → переписка (реплика клиента = исходный текст, реплики оператора — шаблоны) → (итог) → ПД."""
    person = {"link": link(rng), "name": fio(rng), "phone": phone(rng), "address": address(rng), "email": email(rng),
              "account": str(rng.randrange(100000, 9999999))}
    head = rng.choice([person["link"], f"{rng.choice(CHANNELS)}, скрин во вложении\n{person['link']}"])
    lines = [head, "Переписка:", f"Мы: {rng.choice(GREETINGS)}" if rng.random() < 0.3 else None, f"Клиент: {text}",
             f"Мы: {rng.choice(GREETINGS)}", f"Клиент: {rng.choice(['написал в лс', person['phone'] + ', перезвоните', 'жду', 'договор ' + person['account']])}",
             f"Мы: {rng.choice(REPLIES)}"]
    if summary:
        lines.append(f"Итог: {summary}")
    pd_line = rng.choice(["{name}, тел. {phone}, {address}, л/с {account}", "ФИО: {name}\nТелефон: {phone}\nАдрес: {address}",
                          "контакт: {phone} ({name}), {email}, л/с {account}", "{name} {phone} {address}"]).format(**person)
    lines.append(pd_line)
    return Dirty("\n".join(l for l in lines if l), "smm", person)


ALT_GREETINGS = ["Добрый день! Меня зовут {op}, я помогу разобраться. Опишите, пожалуйста, ситуацию подробнее.",
                 "Здравствуйте! Благодарим, что написали нам. Уточните, пожалуйста, адрес и номер обращения.",
                 "Приветствуем! Сожалеем, что так вышло. Проверим и вернёмся с ответом."]
ALT_REPLIES = ["Информация принята, обращение передано ответственному сотруднику, срок ответа — 3 рабочих дня.",
               "Мы зафиксировали ваш отзыв и направили его руководству для разбора.",
               "Спасибо за ожидание! Вопрос решён, если появятся вопросы — пишите в любое время."]
ALT_SUMMARIES = ["Клиенту дана консультация, обращение закрыто.", "Жалоба передана во второй уровень поддержки.",
                 "Отзыв зарегистрирован, клиент уведомлён о сроках.", "Требуется обратная связь от профильного отдела."]
ALT_CHANNELS = ["ВКонтакте", "Telegram", "Одноклассники", "e-mail", "отзыв на сайте"]


def alt_name(rng: random.Random) -> str:
    male = rng.random() < 0.5
    sur = rng.choice(SURNAMES) + ("" if male else "а")
    first = rng.choice(FIRST_M if male else FIRST_F)
    patr = rng.choice(PATR_M if male else PATR_F)
    return rng.choice([f"{sur} {first[0]}. {patr[0]}.", f"{first} {patr} {sur}", f"{first.lower()} {sur.lower()}",
                       f"ув. {first} {patr}", f"@{_translit(first)}_{_translit(sur)}{rng.randrange(10, 99)}"])


def alt_phone(rng: random.Random) -> str:
    a, b, c, d = rng.randrange(900, 999), rng.randrange(100, 999), rng.randrange(10, 99), rng.randrange(10, 99)
    return rng.choice([f"+7 ({a}) {b}-{c}-{d}", f"8 ({a}) {b} {c} {d}", f"8-{a}-{b}-{c}-{d}", f"+7 {a} {b} {c} {d}",
                       f"{a}-{b}-{c}-{d}", f"моб. 8{a}{b}{c}{d}"])


def alt_link(rng: random.Random) -> str:
    n, code = rng.randrange(10 ** 6, 10 ** 9), "".join(rng.choice("abcdefghijkmnpqrstuvwxyzABCDEFGH23456789") for _ in range(6))
    return rng.choice([f"vk.com/id{n}", f"m.vk.com/wall-{n}_{rng.randrange(100, 9999)}", f"www.otzovik.com/review_{n}.html",
                       f"clck.ru/{code}", f"t.me/{code.lower()}_support", f"instagram.com/p/{code}/", f"vk.com/gim{n}?sel={n // 7}"])


def alt_email(rng: random.Random) -> str:
    return (f"{_translit(rng.choice(FIRST_M + FIRST_F))}.{_translit(rng.choice(SURNAMES))}{rng.randrange(1, 99)}"
            f"@{rng.choice(['bk.ru', 'inbox.ru', 'rambler.ru', 'yandex.ru'])}")


def dirty_alt(text: str, rng: random.Random, crm_share: float) -> Dirty:
    """Мусор в форматах, которых не видела очистка (см. уровень alt)."""
    person = {"link": alt_link(rng), "name": alt_name(rng), "phone": alt_phone(rng), "email": alt_email(rng),
              "ticket": str(rng.randrange(100000, 999999)),
              "date": f"{rng.randrange(1, 28):02d}.{rng.randrange(1, 12):02d}.2025 {rng.randrange(8, 22)}:{rng.randrange(10, 59)}"}
    if rng.random() >= crm_share:
        pieces = [person["link"], person["name"], person["phone"], rng.choice([person["email"], f"обращение №{person['ticket']}",
                                                                                person["date"]])]
        sentences = [x for x in _split_sentences(text) if x]
        for piece in rng.sample(pieces, len(pieces)):
            sentences.insert(rng.randrange(0, len(sentences) + 1), piece)
        return Dirty(rng.choice([" ", "\n", "; "]).join(sentences), "alt", person)
    op = rng.choice(FIRST_F + FIRST_M)
    h, m = rng.randrange(8, 21), rng.randrange(10, 50)
    lines = [f"Обращение №{person['ticket']} от {person['date']}", f"Канал: {rng.choice(ALT_CHANNELS)}",
             f"Ссылка на диалог: {person['link']}", "--- Диалог ---", f"[{h}:{m:02d}] Пользователь: {text}",
             f"[{h}:{m + 3:02d}] Сотрудник поддержки ({op}): {rng.choice(ALT_GREETINGS).format(op=op)}",
             f"[{h}:{m + 5:02d}] Пользователь: {rng.choice(['ок', 'спасибо, жду', 'адрес скинул в лс', person['phone']])}",
             f"[{h}:{m + 9:02d}] Сотрудник поддержки ({op}): {rng.choice(ALT_REPLIES)}",
             "--- Резюме ---", rng.choice(ALT_SUMMARIES),
             f"Контактные данные: {person['name']}; {person['phone']}; e-mail: {person['email']}",
             f"Статус: закрыто | Тег: #{rng.choice(['жалоба', 'отзыв', 'вопрос'])}"]
    return Dirty("\n".join(lines), "crm", person)


def _translit(s: str) -> str:
    t = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя", ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n",
                                                      "o", "p", "r", "s", "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y",
                                                      "", "e", "yu", "ya"]))
    return "".join(t.get(ch, ch) for ch in s.lower())


def make_dirty(texts: Sequence[str], level: str = "tz", seed: int = 0, smm_share: float = 0.4,
               summaries: Optional[Sequence[Optional[str]]] = None) -> list[Dirty]:
    rng = random.Random(seed)
    out = []
    for i, t in enumerate(texts):
        s = summaries[i] if summaries is not None else None
        if level == "tz" or (level == "mix" and rng.random() >= smm_share):
            out.append(dirty_tz(t, rng))
        elif level in ("smm", "mix"):
            out.append(dirty_smm(t, rng, s))
        elif level == "alt":
            out.append(dirty_alt(t, rng, smm_share))
        else:
            raise ValueError(f"неизвестный уровень мусора: {level}")
    return out


def _split_sentences(text: str) -> list[str]:
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip())]
