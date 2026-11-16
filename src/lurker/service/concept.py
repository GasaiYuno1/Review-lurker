"""Понятие из промпта → отбор обращений по смыслу. Ядро «универсального» сценария ТЗ:
«нас называют ебланами?» → предклассификация по смыслу → выборка → динамика, тематики, ассистент.

Без LLM: из промпта убираются период и служебные слова → предмет запроса («оскорбляли угрожали физической расправой»);
прототип = эмбеддинг предмета и примеров + псевдо-примеры (обращения со всеми словами предмета, ближайшие к прототипу);
семантическое расширение — соседи слов предмета по векторам navec в словаре корпуса (оскорблять → обзывать, нецензурно).
Пресеты типовых понятий (негатив, оскорбления и мат, благодарность, отток) добавляют примеры и ключевые слова
тому же механизму — это семена, а не классификатор. Оценка обращения = косинус к прототипу + бонус за долю слов предмета
(квадратично: все слова → полный бонус) + бонус за слова пресета + бонус за соседей. Порог — из конфига, в интерфейсе ползунок.
С LLM: критерии (метки с описанием, примерами, ключевыми словами) → те же понятия, несколько сразу.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

import numpy as np

from .. import text as T
from ..period import Period, strip_periods
from ..preclass import Criteria, LabelSpec, _keyword_mask

PRESETS: dict[str, dict] = {
    "оскорбления": {
        "aliases": ["оскорбление", "оскорблять", "хамить", "хамство", "мат", "материться", "ругаться", "ругань", "обзывать",
                    "грубость", "агрессия", "угроза", "угрожать", "расправа", "конфликт", "хуесосить"],
        "examples": ["абонент оскорблял оператора, нецензурно выражался, бросил трубку",
                     "угрожал физической расправой сотрудникам офиса",
                     "кричал, обзывал мошенниками и ворами, обещал приехать в офис и разобраться",
                     "обещал найти мастера и набить морду, угрожал"],
        "keywords": ["оскорблять", "оскорбление", "угрожать", "угроза", "расправа", "нецензурно", "мат", "матом", "хамить",
                     "хамство", "обзывать", "мошенник", "вор", "орать", "морда"],
    },
    "негатив": {
        "aliases": ["негатив", "негативный", "недовольство", "недовольный", "раздражение", "возмущение", "злость", "претензия", "скандал"],
        "examples": ["абонент недоволен, раздражён, сказал что напишет жалобу",
                     "кричал, требовал руководителя, настроен агрессивно",
                     "жалуется на грубость оператора, мастер нахамил",
                     "возмущён двойным списанием, грозит прокуратурой, обзывал мошенниками"],
        "keywords": ["недовольный", "недоволен", "раздражённый", "раздражён", "возмущён", "возмущаться", "агрессивно",
                     "кричать", "орать", "грубость", "хамить", "прокуратура", "обман", "мошенник", "угрожать", "оскорблять",
                     "ужасно", "безобразие", "скандал", "ругаться", "нецензурно"],
    },
    "благодарность": {
        "aliases": ["благодарность", "благодарить", "спасибо", "похвала", "хвалить", "довольный", "позитив", "положительный"],
        "examples": ["благодарит мастера за быстрое подключение", "звонил сказать спасибо, всё починили",
                     "довольна работой техподдержки, просил передать спасибо"],
        "keywords": ["благодарить", "благодарность", "спасибо", "хвалить", "довольный", "довольна", "доволен", "молодец"],
    },
    "отток": {
        "aliases": ["отток", "расторжение", "расторгнуть", "отключиться", "отключение", "уйти", "уходить", "конкурент", "отказаться"],
        "examples": ["хочет расторгнуть договор, уходит к конкуренту, у них дешевле", "просит отключить, переезжает",
                     "хочет отключиться, недоволен качеством"],
        "keywords": ["расторгнуть", "расторжение", "отключить", "отключиться", "конкурент", "дешевле", "уходить", "отказаться"],
    },
}
OBSCENE_RE = re.compile(r"\b(?:ху[йеёия]\w*|пизд\w*|бля\w*|[её]б[аи]\w*|[её]бл\w*|за[её]б\w*|срать|сра[лн]\w*|говн\w*|дерьм\w*|"
                        r"сук[аи]\w*|муда\w*|долбо\w*|еблан\w*)\b", re.I)


@dataclass
class Concept:
    name: str
    subject: str                                   # предмет запроса словами
    core: list[str]                                # леммы предмета
    expanded: list[str] = field(default_factory=list)   # семантические соседи из корпуса
    preset_keywords: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    presets: list[str] = field(default_factory=list)
    proto: Optional[np.ndarray] = None

    @property
    def keywords(self) -> list[str]:
        out = list(self.core)
        for w in self.preset_keywords + self.expanded:
            if w not in out:
                out.append(w)
        return out


def subject_of(prompt: str, ref: date) -> tuple[str, list[Period], list[str]]:
    """Предмет запроса (без периода и служебных слов), периоды, леммы предмета."""
    rest, periods = strip_periods(prompt, ref)
    words = [w for w in T.tokens(rest) if len(w) > 1 and not w.isdigit()]
    core: list[str] = []
    keep = []
    for w in words:
        l = T.lemma(w).replace("ё", "е")
        if l in T.STOPWORDS or l in T.QUERY_STOP or w in T.QUERY_STOP or l in T.DOMAIN_STOP:
            continue
        keep.append(w)
        if l not in core:
            core.append(l)
    return " ".join(keep), periods, core


def match_presets(prompt: str, core: Sequence[str]) -> list[str]:
    """Пресеты, задетые запросом: по леммам предмета и по матерным словам (→ оскорбления и негатив)."""
    hit = []
    for name, p in PRESETS.items():
        if name in core or any(a in core for a in p["aliases"]):
            hit.append(name)
    if OBSCENE_RE.search(prompt.lower().replace("ё", "е")):
        for name in ("оскорбления", "негатив"):
            if name not in hit:
                hit.append(name)
    return hit


def keyword_fraction(texts: Sequence[str], keywords: Sequence[str]) -> np.ndarray:
    """Доля ключевых слов (лемм) запроса, встретившихся в тексте: 0…1."""
    if not keywords:
        return np.zeros(len(texts), dtype=np.float32)
    kw = [{T.lemma(k.replace("ё", "е")), k.replace("ё", "е")} for k in keywords]
    out = np.zeros(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = set(T.tokens(t))
        lem = {T.lemma(x) for x in toks} | toks
        out[i] = sum(1 for k in kw if k & lem) / len(kw)
    return out


def expand_keywords(core: Sequence[str], embedder, top: int = 4, min_sim: float = 0.5, max_total: int = 16) -> list[str]:
    """Соседи слов предмета по navec среди слов корпуса (IDF-словарь NavecEmbedder)."""
    nav = getattr(embedder, "navec", None)
    if nav is None or not hasattr(nav, "idf"):
        return []
    navec = nav.navec
    vocab = [w for w in nav.idf if w in navec and w not in T.STOPWORDS and len(w) > 2]
    if not vocab:
        return []
    M = getattr(nav, "_vocab_matrix", None)
    if M is None or getattr(nav, "_vocab_words", None) != vocab:
        M = np.stack([navec[w] for w in vocab]).astype(np.float32)
        M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-6)
        nav._vocab_matrix, nav._vocab_words = M, vocab
    out: list[str] = []
    for w in core:
        if w not in navec:
            continue
        v = np.asarray(navec[w], dtype=np.float32)
        v /= max(np.linalg.norm(v), 1e-6)
        sims = M @ v
        for j in np.argsort(-sims)[: top + 1]:
            cand = vocab[j]
            if (cand != w and sims[j] >= min_sim and cand not in core and cand not in out
                    and cand not in T.QUERY_STOP and cand not in T.DOMAIN_STOP):
                out.append(cand)
    return out[:max_total]


def prototype(c: Concept, embedder, texts: Sequence[str], E: np.ndarray, max_pseudo: int = 100) -> Optional[np.ndarray]:
    """Среднее эмбеддингов предмета и примеров + псевдо-примеры: обращения со словами предмета (или пресета),
    ближайшие к этому среднему — чтобы прототип не расплывался на всё, где встретилось одно слово."""
    seeds = [s for s in [c.subject] + c.examples if s]
    seed_vec = None
    if seeds:
        seed_vec = embedder.encode(seeds).mean(axis=0)
        seed_vec = seed_vec / (np.linalg.norm(seed_vec) or 1.0)
    if not len(texts):
        return seed_vec
    cand = np.zeros(len(texts), dtype=bool)
    if c.core:
        frac = keyword_fraction(texts, c.core)
        cand = frac >= 0.99
        if cand.sum() < 5:
            cand = frac >= 0.5
    if cand.sum() < 5 and c.preset_keywords:
        cand = cand | _keyword_mask(texts, c.preset_keywords)
    idx = np.where(cand)[0]
    vecs = [seed_vec] if seed_vec is not None else []
    if len(idx):
        if seed_vec is not None and len(idx) > max_pseudo:
            sims = E[idx] @ seed_vec
            idx = idx[np.argsort(-sims)[:max_pseudo]]
        elif len(idx) > max_pseudo:
            idx = np.random.default_rng(0).choice(idx, max_pseudo, replace=False)
        vecs.append(E[idx].mean(axis=0))
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    return v / (np.linalg.norm(v) or 1.0)


def build_concept(prompt: str, embedder, texts: Sequence[str], E: np.ndarray, ref: date,
                  max_pseudo: int = 100, expand: bool = True) -> tuple[Concept, list[Period]]:
    subject, periods, core = subject_of(prompt, ref)
    presets = match_presets(prompt, core)
    examples, preset_kw = [], []
    for name in presets:
        examples += PRESETS[name]["examples"]
        preset_kw += [k for k in PRESETS[name]["keywords"] if k not in preset_kw]
    expanded = expand_keywords(core, embedder) if (expand and core) else []
    name = subject or (", ".join(presets) if presets else prompt.strip())
    c = Concept(name=name, subject=subject, core=core, expanded=expanded, preset_keywords=preset_kw,
                examples=examples, presets=presets)
    c.proto = prototype(c, embedder, texts, E, max_pseudo=max_pseudo)
    return c, periods


def concept_from_label(lab: LabelSpec, embedder, texts: Sequence[str], E: np.ndarray, expand: bool = True) -> Concept:
    """Метка критериев → понятие по схеме пресета: ключевые слова и слова названия — альтернативы (бонус за любое,
    только при семантической близости), примеры и описание — семена прототипа. Слова описания не требуются все сразу:
    описания у LLM общие («негативная реакция абонента на действия компании»)."""
    kws: list[str] = []
    for k in list(lab.keywords) + T.lemmas(lab.name):
        l = T.lemma(k.lower().replace("ё", "е"))
        if l and l not in kws and l not in T.QUERY_STOP and l not in T.STOPWORDS:
            kws.append(l)
    expanded = expand_keywords(kws, embedder) if expand else []
    subject = "; ".join(s for s in [lab.name, lab.description] if s)
    c = Concept(lab.name, subject, [], expanded, preset_keywords=kws, examples=list(lab.examples))
    c.proto = prototype(c, embedder, texts, E)
    return c


def augment_concept(c: Concept, criteria: Criteria, embedder, texts: Sequence[str], E: np.ndarray, max_labels: int = 1) -> Concept:
    """Дополнить понятие из промпта критериями LLM: ключевые слова и примеры первых max_labels меток идут в пресетные
    слова и семена прототипа; предмет запроса остаётся главным сигналом. Прототип пересчитывается."""
    for lab in criteria.labels[:max_labels]:
        for k in list(lab.keywords) + T.lemmas(lab.name):
            l = T.lemma(k.lower().replace("ё", "е"))
            if l and l not in c.preset_keywords and l not in c.core and l not in T.QUERY_STOP and l not in T.STOPWORDS:
                c.preset_keywords.append(l)
        for ex in lab.examples:
            if ex and ex not in c.examples:
                c.examples.append(ex)
    c.proto = prototype(c, embedder, texts, E)
    return c


def criteria_to_concepts(criteria: Criteria, embedder, texts: Sequence[str], E: np.ndarray) -> list[Concept]:
    return [concept_from_label(l, embedder, texts, E) for l in criteria.labels]


def score(c: Concept, texts: Sequence[str], E: np.ndarray, core_bonus: float = 0.25, preset_bonus: float = 0.15,
          expanded_bonus: float = 0.08, min_cos: float = 0.25) -> np.ndarray:
    """Косинус к прототипу + бонусы. Слова предмета запроса — безусловно (квадратично по доле совпавших);
    слова пресета и соседи — только если текст уже семантически близок (косинус ≥ min_cos): они уточняют выборку, а не создают её."""
    cos = (np.asarray(E) @ c.proto) if c.proto is not None else np.zeros(len(texts), dtype=np.float32)
    s = cos.copy()
    if c.core:
        s = s + core_bonus * keyword_fraction(texts, c.core) ** 2
    near = cos >= min_cos
    if c.preset_keywords:
        s = s + preset_bonus * (_keyword_mask(texts, c.preset_keywords) & near)
    if c.expanded:
        s = s + expanded_bonus * (_keyword_mask(texts, c.expanded) & near)
    return s.astype(np.float32)


def assign(concepts: Sequence[Concept], texts: Sequence[str], E: np.ndarray, threshold: float,
           other: str = "прочее") -> tuple[list[str], np.ndarray]:
    """Несколько понятий: каждому обращению — понятие с максимальной оценкой, если она ≥ порога."""
    S = np.stack([score(c, texts, E) for c in concepts], axis=1) if concepts else np.zeros((len(texts), 0))
    labels = []
    for row in S:
        j = int(np.argmax(row)) if len(row) else -1
        labels.append(concepts[j].name if j >= 0 and row[j] >= threshold else other)
    return labels, S
