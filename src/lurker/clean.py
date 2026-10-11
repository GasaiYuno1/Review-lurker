"""Очистка текста перед эмбеддингом (ТЗ п. 4). Сырой текст в эмбеддинг не идёт.

Слой 1 — правила (`RuleCleaner`, без LLM):
  * структура SMM «ссылка → переписка → саммари → ПД»: берётся саммари; если его нет — реплики клиента;
  * URL, email, телефоны, лицевые счета/договоры/номера заявок, адреса, ФИО (регулярки + NER natasha) удаляются.
Слой 2 — LLM (`LLMCleaner`): few-shot, батчами, JSON «номер → чистый текст». На вход подаётся результат слоя 1:
  дешевле по токенам, и персональные данные не уходят во внешний API. Сбой LLM на батче → остаётся результат слоя 1.

`clean_texts()` чистит список текстов с кэшем по хэшу текста (дубли чистятся один раз).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

# ---------- регулярки ----------
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:ru|com|net|org|io|me|su|рф)(?:/\S*)?", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?7|8)[\s(-]*9\d{2}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)|(?<!\d)(?<!\d[.,])\d{9,12}(?!\d)(?![.,]\d)"
                      r"|(?<![\d+])\(?9\d{2}\)?[\s-]\d{3}[\s-]\d{2}[\s-]\d{2}(?!\d)")        # «953-619-59-83» без кода страны
HANDLE_RE = re.compile(r"(?<![\w.])@[A-Za-z0-9_.]{3,}")                              # ник «@ivan_petrov88»
ACCOUNT_RE = re.compile(r"(?:л/с|лс|лицев\w*\s+сч\w*|договор\w*|дог\.|заявк\w*|номер)\s*(?:№|#|n|:)?\s*\d{4,}", re.I)
NUMBER_TAG_RE = re.compile(r"(?:№|#)\s*\d+")
TIMESTAMP_RE = re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]")                       # «[14:02]» в выгрузках чатов
ADDRESS_RE = re.compile(
    r"(?:\bадрес\s*:\s*)?\b(?:ул\.|улица|пр-т\.?|пр\.|проспект|пер\.|переулок|б-р|бульвар|ш\.|шоссе|пл\.|мкр\.?|микрорайон)\s*"
    r"[А-ЯЁа-яё-]+(?:\s+[А-ЯЁа-яё-]+)?(?:\s*,?\s*(?:\bд\.|\bдом\b)\s*\d+\s*[а-яё]?)?(?:\s*,?\s*(?:\bкорп\.|\bк\.)\s*\d+)?"
    r"(?:\s*,?\s*(?:\bкв\.|\bквартира\b|\bоф\.|\bофис\b)\s*\d+)?", re.I)
ADDRESS_SHORT_RE = re.compile(r"\b[А-ЯЁ][а-яё]+\s+\d{1,3}-\d{1,4}\b")            # «Лесная 28-149»
ADDRESS_LABEL_RE = re.compile(r"адрес\s*:\s*[^\n|;]{1,60}?(?=[\n|;]|\s{2}|$|,\s*(?:тел|кв|л/с|\+|8\d|email))", re.I)
# метка + само имя (1–3 слова с заглавной, инициалы), не «всё до конца строки»: в однострочном тексте это съедало весь отзыв
FIO_LABEL_RE = re.compile(r"\b(?:фио|ф\.и\.о\.?|имя|контакт|автор|клиент)\s*:\s*(?:[А-ЯЁ][а-яё]+(?:\s+|$)){0,3}(?:[А-ЯЁ]\.\s?){0,2}", re.I)
PATR = r"(?:ович|евич|ьич|ич|овна|евна|ична|инична)"
FIO_FULL_RE = re.compile(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+" + PATR + r"\b")      # Фамилия Имя Отчество
FIO_NAME_PATR_RE = re.compile(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+" + PATR + r"\b")                 # Имя Отчество
FIO_INITIALS_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\s*[А-ЯЁ]\.\s?[А-ЯЁ]\.|\b[А-ЯЁ]\.\s?[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]{2,}\b")  # Иванова Е.С. / Е.С. Иванова
FIO_PAREN_RE = re.compile(r"\(\s*[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}\s*\)")
SURNAME_ENDINGS = r"(?:ов|ев|ёв|ин|ын|ова|ева|ёва|ина|ына|ский|ская|цкий|цкая|ко|ук|юк|ич|енко)"
FIO_SURNAME_FIRST_RE = re.compile(r"\b[А-ЯЁ][а-яё]+" + SURNAME_ENDINGS + r"\s+[А-ЯЁ][а-яё]{2,}\b")   # Иванов Пётр (без отчества)

SUMMARY_RE = re.compile(r"(?:итог|саммари|резюме|summary|суть|результат)\s*:\s*(.+?)(?=\n|\s*\|\s*|$)", re.I | re.S)
OUR_LINE_RE = re.compile(r"(?:^|\n)\s*(?:мы|оператор|поддержка|we)\s*:.*", re.I)
SPEAKER_RE = re.compile(r"(?:^|\n|\|\s*|\s/\s*)\s*(клиент|абонент|пользователь|client|мы|оператор|поддержка|we)\s*:", re.I)
CLIENT_SPEAKERS = {"клиент", "абонент", "пользователь", "client"}


def client_turns(text: str) -> list[str]:
    """Реплики клиента в переписке: от «Клиент:» до следующего говорящего (многострочные реплики целиком).
    Хвост последней реплики обрезается на «Итог:/Саммари:» и строках-разделителях «|»."""
    marks = list(SPEAKER_RE.finditer(text))
    out = []
    for i, m in enumerate(marks):
        if m.group(1).lower() not in CLIENT_SPEAKERS:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        seg = text[m.end():end]
        seg = re.split(r"\n\s*(?:итог|саммари|резюме)\s*:|\s\|\s", seg, maxsplit=1, flags=re.I)[0]
        seg = seg.strip()
        if seg:
            out.append(seg)
    return out
NOISE_RE = re.compile(r"\bпереписка\s*:|\bскрин\w*\s+во\s+вложении|\bво\s+вложении|\bличка\s+вк\b|\bдирект\b|\bкомментарий\s+под\s+постом|"
                      r"\bотзыв\s+на\s+картах|\bчат\s+поддержки\b|\bтел\.|\bтел\s*:|\bтелефон\s*:|\bдоговор\s*:|\bл/с\b|\bлс\b|\bконтакт\b\s*:?|"
                      r"\bфио\s*:|\bавтор\s*:|\bклиент\s*:", re.I)
SPACE_RE = re.compile(r"[ \t]+")
PUNCT_RUN_RE = re.compile(r"\s*([,;|/–—-]\s*){2,}")
EDGE_PUNCT_RE = re.compile(r"^[\s,;|/–—.:-]+|[\s,;|/–—:-]+$")

PLACEHOLDERS = {"url": "[ссылка]", "email": "[email]", "phone": "[телефон]", "account": "[номер]", "address": "[адрес]",
                "name": "[имя]", "number": "[номер]"}


@dataclass
class CleanResult:
    text: str
    removed: Counter = field(default_factory=Counter)
    structure: str = "plain"     # plain | summary | client_lines


@lru_cache(maxsize=200_000)
def _name_kind(word: str) -> str:
    """«Name» / «Surn» / "" по первому разбору pymorphy (кэш: слова повторяются)."""
    tag = RuleCleaner._morph.parse(word)[0].tag
    return "Name" if "Name" in tag else "Surn" if "Surn" in tag else ""


class RuleCleaner:
    """Очистка правилами. ner=True подключает NER natasha (PER/LOC) поверх регулярок, если пакет доступен."""

    def __init__(self, ner: bool = True, placeholders: bool = False):
        self.placeholders = placeholders
        self._ner = None
        if ner:
            try:
                from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
                self._ner = (Segmenter(), NewsNERTagger(NewsEmbedding()), Doc)
            except Exception:
                self._ner = None

    @property
    def name(self) -> str:
        return "rules+ner" if self._ner else "rules"

    def clean(self, text: str) -> CleanResult:
        text = (text or "").replace("\r", "")
        res = self._clean(text, extract=True)
        if res.structure != "plain" and len(res.text) < 10:
            # структура распознана ошибочно (например, «Клиент: Иван Петров» в конце обычного текста) — чистим целиком
            full = self._clean(text, extract=False)
            if len(full.text) > len(res.text):
                return full
        return res

    def _clean(self, text: str, extract: bool) -> CleanResult:
        removed: Counter = Counter()
        structure = "plain"
        if extract:
            text, structure = self._extract_content(text, removed)

        def sub(pattern: re.Pattern, kind: str, s: str) -> str:
            n = len(pattern.findall(s))
            if n:
                removed[kind] += n
                s = pattern.sub(f" {PLACEHOLDERS[kind]} " if self.placeholders else " ", s)
            return s

        text = sub(EMAIL_RE, "email", text)          # раньше ссылок: иначе домен «yandex.ru» съедается как ссылка, остаётся «ivan.petrov@»
        text = sub(URL_RE, "url", text)
        text = sub(TIMESTAMP_RE, "number", text)
        text = sub(PHONE_RE, "phone", text)
        text = sub(FIO_LABEL_RE, "name", text)
        text = sub(ADDRESS_LABEL_RE, "address", text)
        text = sub(ADDRESS_RE, "address", text)
        text = sub(ADDRESS_SHORT_RE, "address", text)
        text = sub(ACCOUNT_RE, "account", text)
        text = sub(NUMBER_TAG_RE, "number", text)
        text = sub(FIO_FULL_RE, "name", text)
        text = sub(FIO_NAME_PATR_RE, "name", text)
        text = sub(FIO_INITIALS_RE, "name", text)
        text = sub(FIO_PAREN_RE, "name", text)
        text = sub(HANDLE_RE, "name", text)
        text = self._lower_names(text, removed)
        if self._ner:
            text = self._ner_strip(text, removed)
        else:
            text = sub(FIO_SURNAME_FIRST_RE, "name", text)
        text = NOISE_RE.sub(" ", text)
        text = self._tidy(text)
        return CleanResult(text, removed, structure)

    # ---- структура SMM ----
    @staticmethod
    def _looks_structured(text: str) -> bool:
        """Признаки тикета-переписки: реплики «Мы:/Оператор:», «Переписка:», ссылка в первой строке.
        Без них «Итог:» или «Клиент:» внутри обычного отзыва не считаются структурой."""
        return bool(OUR_LINE_RE.search(text) or re.search(r"переписка\s*:|(?:^|\s)(?:мы|оператор)\s*:|\n\s*[—–-]\s+\S", text, re.I))

    def _extract_content(self, text: str, removed: Counter) -> tuple[str, str]:
        if not self._looks_structured(text):
            return text, "plain"
        m = SUMMARY_RE.search(text)
        if m:
            removed["structure_summary"] += 1
            return m.group(1), "summary"
        client = client_turns(text)
        if client:
            removed["structure_client"] += 1
            return " ".join(client), "client_lines"
        if OUR_LINE_RE.search(text):
            text = OUR_LINE_RE.sub(" ", text)
        return text, "plain"

    _morph = None

    def _lower_names(self, text: str, removed: Counter) -> str:
        """Имя и фамилия строчными подряд («алексей лебедев»): NER опирается на заглавные буквы и их не видит.
        Пара — только «имя + фамилия» или «фамилия + имя» по словарю pymorphy (одиночные «вера», «роза» не трогаются)."""
        if RuleCleaner._morph is None:
            try:
                import pymorphy3
                RuleCleaner._morph = pymorphy3.MorphAnalyzer()
            except Exception:
                RuleCleaner._morph = False
        if not RuleCleaner._morph:
            return text
        toks = list(re.finditer(r"\b[а-яё]{3,}\b", text))
        cut = []
        for a, b in zip(toks, toks[1:]):
            if text[a.end():b.start()].strip():
                continue
            ta, tb = _name_kind(a.group()), _name_kind(b.group())
            if {ta, tb} == {"Name", "Surn"}:
                cut.append((a.start(), b.end()))
        for st, en in reversed(cut):
            removed["name"] += 1
            text = text[:st] + (f" {PLACEHOLDERS['name']} " if self.placeholders else " ") + text[en:]
        return text

    def _ner_strip(self, text: str, removed: Counter) -> str:
        seg, tagger, Doc = self._ner
        doc = Doc(text)
        doc.segment(seg)
        try:
            doc.tag_ner(tagger)
        except Exception:
            return text
        spans = [s for s in doc.spans if s.type in ("PER", "LOC")]
        if not spans:
            return text
        out, pos = [], 0
        for s in sorted(spans, key=lambda s: s.start):
            out.append(text[pos:s.start])
            kind = "name" if s.type == "PER" else "address"
            removed[kind] += 1
            out.append(f" {PLACEHOLDERS[kind]} " if self.placeholders else " ")
            pos = s.stop
        out.append(text[pos:])
        return "".join(out)

    @staticmethod
    def _tidy(s: str) -> str:
        s = s.replace("\n", " ").replace("|", " ")
        s = SPACE_RE.sub(" ", s)
        s = PUNCT_RUN_RE.sub(lambda m: m.group(1).strip() + " ", s)
        s = re.sub(r"\s+([,.;!?])", r"\1", s)
        s = re.sub(r"\(\s*\)", " ", s)
        s = re.sub(r"\.(\s*\.)+", ".", s)
        s = SPACE_RE.sub(" ", s)
        return EDGE_PUNCT_RE.sub("", s).strip()


# ---------- LLM few-shot ----------

CLEAN_SYSTEM = """Ты готовишь комментарии сотрудников поддержки к семантическому анализу.
Из каждого комментария оставь только содержательную часть: что случилось у абонента и что сделали.
Убери ссылки, переписку с клиентом (оставь её суть одной фразой), персональные данные (ФИО, телефоны, адреса, номера договоров),
служебную разметку («Итог:», «Переписка:», номера заявок), приветствия и вежливые формулы. Сокращения раскрой (инет → интернет, тп → техподдержка).
Не добавляй ничего от себя, не оценивай. Пустой или бессодержательный комментарий → пустая строка.
Отвечай строго JSON-объектом: ключ — номер комментария, значение — очищенный текст."""

CLEAN_EXAMPLES = [
    ("https://vk.com/wall-123_456?reply=7\nПереписка:\nКлиент: Домофон перестал звонить в приложение! Курьер стоял 10 минут\n"
     "Мы: Здравствуйте! Подскажите адрес и номер договора\nКлиент: Ленина 5-12, договор 123456\nМы: Передали специалистам\n"
     "Итог: не приходит звонок с домофона в приложение, попросили обновить приложение\nИванова Мария Петровна, тел. +7 925 123-45-67",
     "не приходит звонок с домофона в приложение, попросили обновить приложение"),
    ("абон: нет инет, ребутнули роутер не помогло, заяв на выезд, абон недоволен",
     "нет интернета, перезагрузка роутера не помогла, заявка на выезд, абонент недоволен"),
    ("Пропали каналы. Приставка на старой прошивке. Пересканировали каналы, помогло. Заявка №367422.",
     "пропали каналы, приставка на старой прошивке, пересканировали каналы, помогло"),
    ("контакт: 89184198200 (Соловьёва Екатерина), л/с 6765535 | клиент: как заказать ключи от домофона? / мы: напишите адрес | саммари: заказ ключей, 3 шт, оформили",
     "заказ ключей от домофона, 3 шт, оформили"),
    ("-", ""),
]


def build_clean_messages(batch: Sequence[str]) -> list[dict]:
    ex_in = {str(i + 1): raw for i, (raw, _) in enumerate(CLEAN_EXAMPLES)}
    ex_out = {str(i + 1): clean for i, (_, clean) in enumerate(CLEAN_EXAMPLES)}
    return [
        {"role": "system", "content": CLEAN_SYSTEM},
        {"role": "user", "content": json.dumps(ex_in, ensure_ascii=False, indent=0)},
        {"role": "assistant", "content": json.dumps(ex_out, ensure_ascii=False, indent=0)},
        {"role": "user", "content": json.dumps({str(i + 1): t for i, t in enumerate(batch)}, ensure_ascii=False, indent=0)},
    ]


class LLMCleaner:
    """Слой 2: LLM few-shot батчами поверх RuleCleaner. Любой объект с методом chat_json(messages)."""

    def __init__(self, llm, batch_size: int = 20, prefilter: Optional[RuleCleaner] = None, max_chars: int = 1500):
        self.llm = llm
        self.batch_size = batch_size
        self.prefilter = prefilter or RuleCleaner()
        self.max_chars = max_chars
        self.failed_batches = 0

    @property
    def name(self) -> str:
        return f"llm:{getattr(getattr(self.llm, 'cfg', None), 'model', 'fake')}+{self.prefilter.name}"

    def clean_many(self, texts: Sequence[str]) -> list[CleanResult]:
        pre = [self.prefilter.clean(t) for t in texts]
        out = list(pre)
        todo = [i for i, r in enumerate(pre) if r.text]
        for k in range(0, len(todo), self.batch_size):
            idx = todo[k:k + self.batch_size]
            batch = [pre[i].text[:self.max_chars] for i in idx]
            try:
                data = self.llm.chat_json(build_clean_messages(batch))
                if not isinstance(data, dict):
                    raise ValueError("не объект")
            except Exception:
                self.failed_batches += 1
                continue
            for j, i in enumerate(idx):
                val = data.get(str(j + 1))
                if isinstance(val, str):
                    r = pre[i]
                    out[i] = CleanResult(RuleCleaner._tidy(val), r.removed + Counter(llm=1), r.structure)
        return out


# ---------- кэш и пакетная очистка ----------

class TextCache:
    """sqlite: ключ (backend + хэш текста) → очищенный текст. Один файл на кэш-каталог."""

    def __init__(self, path: Optional[str | Path]):
        self.conn = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path))
            self.conn.execute("CREATE TABLE IF NOT EXISTS clean (k TEXT PRIMARY KEY, v TEXT)")

    @staticmethod
    def key(backend: str, text: str) -> str:
        return backend + ":" + hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get_many(self, keys: Iterable[str]) -> dict[str, str]:
        if not self.conn:
            return {}
        keys = list(keys)
        out: dict[str, str] = {}
        for k in range(0, len(keys), 500):
            chunk = keys[k:k + 500]
            q = f"SELECT k, v FROM clean WHERE k IN ({','.join('?' * len(chunk))})"
            out.update(dict(self.conn.execute(q, chunk).fetchall()))
        return out

    def put_many(self, items: dict[str, str]) -> None:
        if self.conn and items:
            self.conn.executemany("INSERT OR REPLACE INTO clean VALUES (?, ?)", list(items.items()))
            self.conn.commit()


SEG_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")                 # конец предложения (точка перед пробелом) или строки


def _seg_key(seg: str) -> str:
    s = EMAIL_RE.sub(" e ", seg.lower())
    s = URL_RE.sub(" u ", s)
    s = re.sub(r"\([^)]{0,40}\)", " ", s)                 # «(Анна)» — имя сотрудника в шаблонной строке
    s = re.sub(r"\d+", "0", s)
    return " ".join(re.sub(r"[^\w\s]", " ", s).split())


LINE_LABEL_RE = re.compile(r"^\s*([^:\n]{1,40}?)\s*:(?!//)\s*")
LEAD_TIME_RE = re.compile(r"^\s*\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]\s*")
KNOWN_SPEAKERS = {"клиент", "абонент", "пользователь", "client", "мы", "оператор", "поддержка", "we", "менеджер"}
RULE_LABELS = KNOWN_SPEAKERS | {"фио", "ф и о", "имя", "контакт", "автор", "телефон", "тел", "адрес", "email", "e mail", "почта",
                                "договор", "л с", "лс", "итог", "саммари", "резюме", "summary", "суть", "результат", "переписка"}


def _keep_rule_label(x: str) -> str:
    """Метка, по которой правила разбирают структуру («Мы:», «Клиент:», «Переписка:»), остаётся и у шаблонной строки:
    иначе правила не узнают переписку и примут «Клиент: Очень…» за метку с именем."""
    m = LINE_LABEL_RE.match(x)
    return x[:m.end()] if m and _seg_key(m.group(1)) in RULE_LABELS else ""


def remove_boilerplate(texts: Sequence[str], min_words: int = 4, min_share: float = 0.01, min_docs: int = 5,
                       line_share: float = 0.05) -> tuple[list[str], list[tuple[str, int]]]:
    """Корпусная очистка от шаблонов, не зависящая от формата выгрузки (запускать до правил RuleCleaner).
    1) Предложения и строки из ≥ min_words слов (после нормализации: цифры, ссылки, e-mail, имя в скобках), которые
       повторяются в ≥ min_share документов (не меньше min_docs), — шаблон: ответы операторов, шапки и статусы CRM,
       итоги «консультация оказана»; вырезаются на месте.
    2) Строки: время в начале («[14:02]») снимается; метка строки «…:» (до 4 слов), которая стоит в начале строк в
       ≥ line_share документов, — служебная («Канал:», «Сотрудник поддержки (Анна):», «Контактные данные:») и снимается,
       кроме меток, по которым работают правила (собеседники «Клиент:», «Мы:»…, ПД «ФИО:», «Телефон:», итог); строка,
       которая после этого повторяется в ≥ line_share документов («--- Диалог ---», «Здравствуйте!», «ок»), удаляется.
    Документ, от которого осталось бы меньше трёх слов, не трогается.
    → (тексты, удалённое с числом документов — для проверки глазами)."""
    texts = [t or "" for t in texts]
    n = len(texts)
    segs = [[x for x in SEG_SPLIT_RE.split(t) if x.strip()] for t in texts]
    df: Counter = Counter()
    for ss in segs:
        df.update({k for k in map(_seg_key, ss) if len(k.split()) >= min_words})
    bad = {k for k, c in df.items() if c >= max(min_docs, min_share * n)}
    stage = []
    for t, ss in zip(texts, segs):
        new = t
        for x in ss:
            if _seg_key(x) in bad:
                new = new.replace(x, _keep_rule_label(x) or " ", 1)
        stage.append(new)
    thr2 = max(min_docs, line_share * n)
    lines = [[LEAD_TIME_RE.sub("", ln) for ln in t.split("\n")] for t in stage]

    def label(ln: str) -> Optional[str]:
        m = LINE_LABEL_RE.match(ln)
        k = _seg_key(m.group(1)) if m else ""
        return k if 1 <= len(k.split()) <= 4 else None

    ldf: Counter = Counter()
    for ls in lines:
        ldf.update({k for k in map(label, ls) if k})
    bad_labels = {k for k, c in ldf.items() if c >= thr2 and k not in RULE_LABELS}     # метки, по которым работают правила, — не трогаем
    lines = [[LINE_LABEL_RE.sub("", ln, count=1) if label(ln) in bad_labels else ln for ln in ls] for ls in lines]
    rdf: Counter = Counter()
    for ls in lines:
        rdf.update({k for k in map(_seg_key, ls) if k})
    bad_lines = {k for k, c in rdf.items() if c >= thr2}
    out = []
    for t, ls in zip(texts, lines):
        new = "\n".join(_keep_rule_label(ln) if _seg_key(ln) in bad_lines else ln for ln in ls)
        out.append(new if len(re.findall(r"\w{2,}", new)) >= 3 else t)
    removed = ([(k, df[k]) for k in bad] + [(f"метка: {k}", ldf[k]) for k in bad_labels] +
               [(f"строка: {k}", rdf[k]) for k in bad_lines])
    return out, sorted(removed, key=lambda kv: -kv[1])


def clean_texts(texts: Sequence[str], cleaner=None, cache: Optional[TextCache] = None) -> list[str]:
    """Чистит тексты: одинаковые — один раз, результат кэшируется по хэшу (backend + текст)."""
    cleaner = cleaner or RuleCleaner()
    uniq = list(dict.fromkeys(t or "" for t in texts))
    keys = {t: TextCache.key(cleaner.name, t) for t in uniq}
    done = cache.get_many(keys.values()) if cache else {}
    todo = [t for t in uniq if keys[t] not in done]
    if todo:
        if hasattr(cleaner, "clean_many"):
            results = cleaner.clean_many(todo)
        else:
            results = [cleaner.clean(t) for t in todo]
        fresh = {keys[t]: r.text for t, r in zip(todo, results)}
        done.update(fresh)
        if cache:
            cache.put_many(fresh)
    return [done[keys[t or ""]] for t in texts]
