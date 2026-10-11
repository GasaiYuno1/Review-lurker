#!/usr/bin/env python3
"""Собирает и исполняет ноутбуки исследования из результатов в research/results/ (тяжёлые расчёты — в скриптах,
ноутбук только читает их итоги, поэтому исполняется за минуты). Выводы — из research/conclusions_*.md.
python3 research/make_notebooks.py [--no-exec]
→ research/01_comments_clustering.ipynb, research/02_anomalous_calls.ipynb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = r'''
import json, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "research" else Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "research"))
RES = ROOT / "research" / "results"
DATA = ROOT / "data" / "research"
pd.set_option("display.max_colwidth", 120, "display.width", 200)
# категориальная палитра (фиксированный порядок слотов): исходный / замусоренный / очищенный
C = {"orig": "#2a78d6", "dirty": "#eb6834", "clean": "#1baf7a"}
NAMES = {"orig": "исходный", "dirty": "замусоренный", "clean": "очищенный"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10})

def grouped(ax, table, variants, ylabel, title, ylim=(0, 1)):
    """Сгруппированные столбцы: модели по оси X, варианты текста — цветом (с легендой)."""
    models = list(table.index)
    w = 0.8 / len(variants)
    for j, v in enumerate(variants):
        if v not in table.columns:
            continue
        x = np.arange(len(models)) + (j - (len(variants) - 1) / 2) * w
        ax.bar(x, table[v].values, width=w * 0.92, color=C[v], label=NAMES[v], zorder=3)
    ax.set_xticks(np.arange(len(models)), models, rotation=20, ha="right")
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=len(variants), loc="upper left", bbox_to_anchor=(0, -0.28))
'''


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def conclusions(name: str) -> str:
    p = HERE / f"conclusions_{name}.md"
    return p.read_text(encoding="utf-8") if p.exists() else "_Выводы будут добавлены после расчётов._"


def nb_comments():
    cells = [md("""
# Задача 1. Семантический анализ комментариев: эмбеддинги и кластеризация

**Гипотеза (ТЗ v2):** эмбеддинги и кластеризация помогут выявлять картину негатива и боли в отзывах абонентов —
автоматически, без предварительного обучения, группировать схожие комментарии.

**Что проверяем (вопросы ТЗ):** (1) находятся ли семантически похожие комментарии в одной группе;
(2) сколько нерелевантных комментариев попадает в кластер; (3) можно ли дать кластеру понятное описание;
(4) насколько результат зависит от предобработки текста.

**Как воспроизвести** (тяжёлое — скриптами, ноутбук только читает итоги):
```
python3 research/comments/prepare_data.py        # данные + замусоривание + очистка
python3 research/comments/embed.py               # эмбеддинги (часы на CPU, кэш в build/)
python3 research/comments/experiment.py          # кластеры и метрики -> research/results/comments_metrics.csv
python3 research/comments/sensitivity.py         # настройки UMAP/HDBSCAN
python3 research/comments/describe.py --dataset bank --variant clean --model <лучшая>   # описания кластеров LLM
python3 research/make_notebooks.py               # этот ноутбук
```
"""), code(SETUP), md("""
## 1. Данные

| набор | что это | метка «о чём текст» | зачем |
|---|---|---|---|
| **bank** | 4000 отзывов о банках с banki.ru (Kaggle, датасет из ТЗ) | нет (есть рейтинг 1–5) | ближе всего к «за что ругают и хвалят»; интерпретация кластеров |
| **geo** | 4200 отзывов Яндекс Карт, 12 непохожих рубрик × 350 | рубрика | честный замер «похожие в одной группе» на открытых данных |
| **tickets** | 2984 синтетических тикета, 25 тем | скрытая тема | формат реальных тикетов: короткие, с SMM-мусором |

Варианты текста: **исходный**; **замусоренный** — по ТЗ в каждый текст ссылка, имя и телефон, а 40% текстов переписаны
в формат SMM-тикета (ссылка → переписка с шаблонными репликами оператора → блок ПД); **очищенный** — замусоренный после
очистки правилами + NER (`lurker.clean.RuleCleaner`). У тикетов исходного варианта нет: мусор в них с рождения.
"""), code(r'''
sets = {n: pd.read_parquet(DATA / "comments" / f"{n}.parquet") for n in ("bank", "geo", "tickets")}
pd.DataFrame({n: {"текстов": len(d), "меток": d["label"].nunique() if d["label"].astype(bool).any() else 0,
                  "ср. длина исходного": round(d["orig"].str.len().mean()) if d["orig"].astype(bool).any() else None,
                  "ср. длина замусоренного": round(d["dirty"].str.len().mean()), "ср. длина очищенного": round(d["clean"].str.len().mean())}
              for n, d in sets.items()}).T
'''), code(r'''
b = sets["bank"]
for kind in ("tz", "smm"):
    r = b[b["dirty_kind"] == kind].iloc[3]
    print(f"===== замусоривание «{kind}»\n--- замусоренный:\n{r['dirty'][:700]}\n--- очищенный:\n{r['clean'][:400]}\n")
'''), md("""
## 2. Очистка: сколько мусора снимает и сколько смысла теряет
Сохранность — доля слов исходного отзыва (≥ 4 букв), оставшихся после «замусорить → очистить». Для сравнения — очистка
исходного отзыва без мусора (теряются имена сотрудников, адреса, номера — это тоже персональные данные).
"""), code(r'''
from lurker.clean import RuleCleaner, URL_RE, PHONE_RE, EMAIL_RE
rc = RuleCleaner()
W = lambda x: set(re.findall(r"[а-яё]{4,}", x.lower()))
rows = []
for n in ("bank", "geo"):
    d = sets[n].head(1500)
    keep = [len(W(o) & W(c)) / max(len(W(o)), 1) for o, c in zip(d["orig"], d["clean"])]
    keep0 = [len(W(o) & W(rc.clean(o).text)) / max(len(W(o)), 1) for o in d["orig"]]
    rows.append({"набор": n, "сохранность после мусор→очистка": np.mean(keep), "сохранность очистки чистого текста": np.mean(keep0),
                 "ссылки в очищенном": d["clean"].str.contains(URL_RE).mean(), "телефоны в очищенном": d["clean"].str.contains(PHONE_RE).mean(),
                 "email в очищенном": d["clean"].str.contains(EMAIL_RE).mean()})
pd.DataFrame(rows).set_index("набор").round(3)
'''), md("""
## 3. Как мусор меняет эмбеддинги — независимо от кластеризации
* **соседи сохранились** — доля общих 10 ближайших соседей текста с его исходным вариантом (1 — мусор ничего не изменил);
* **SMM видит SMM** — у текстов в SMM-формате доля соседей тоже в SMM-формате (в корпусе таких 40%: всё выше — модель
  группирует по формату, а не по смыслу);
* **в «форматных» кластерах** — доля текстов в кластерах, где ≥ 80% текстов в SMM-формате.
"""), code(r'''
m = pd.read_csv(RES / "comments_metrics.csv")
order = [x for x in ["tfidf", "hybrid", "rubert-tiny2", "e5-small", "user-base", "e5-base", "user-bge-m3", "frida"] if x in set(m.model)]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, ds in zip(axes, ("bank", "geo")):
    t = m[(m.dataset == ds) & m.variant.isin(["dirty", "clean"])].pivot(index="model", columns="variant", values="knn_overlap").reindex([o for o in order if o in set(m[m.dataset == ds].model)])
    grouped(ax, t, ["dirty", "clean"], "доля общих соседей", f"{ds}: соседство совпадает с исходным текстом")
plt.tight_layout(); plt.show()
m[m.dataset.isin(["bank", "geo"])].pivot_table(index=["dataset", "model"], columns="variant",
    values=["knn_overlap", "smm_nn_share", "format_docs"]).round(3)
'''), md("""
## 4. Попадают ли похожие тексты в одну группу, и сколько в группе чужих
Метки: рубрика (geo), скрытая тема (tickets). **B³-точность** — для каждого текста доля «своих» в его кластере
(1 − B³-точность ≈ «доля нерелевантных в кластере»); **B³-полнота** — доля «своих», попавших в тот же кластер
(собираются ли похожие вместе; низкая — тема раздроблена на подтемы); **чистые кластеры** — доля кластеров с чистотой ≥ 0.8.
Кластеризация: UMAP (15 соседей, 5 измерений, косинус) → HDBSCAN → дораспределение выбросов (схема BERTopic с настройкой).
"""), code(r'''
fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
for row, ds in enumerate(("geo", "tickets")):
    variants = ["orig", "dirty", "clean"] if ds == "geo" else ["dirty", "clean"]
    for col, metric in enumerate(("bcubed_p", "bcubed_r")):
        t = m[m.dataset == ds].pivot(index="model", columns="variant", values=metric).reindex([o for o in order if o in set(m[m.dataset == ds].model)])
        grouped(axes[row, col], t, variants, "B³-" + ("точность" if metric.endswith("p") else "полнота"),
                f"{ds}: B³-{'точность (нет чужих)' if metric.endswith('p') else 'полнота (свои вместе)'}")
plt.tight_layout(); plt.show()
cols = ["n_clusters", "outliers", "nmi", "purity", "bcubed_p", "bcubed_r", "clean_clusters", "km_nmi", "km_bcubed_p", "km_bcubed_r"]
m[m.dataset.isin(["geo", "tickets"])].set_index(["dataset", "model", "variant"])[cols].round(3)
'''), md("""
## 5. Настройка кластеризации
Сравнение настроек UMAP/HDBSCAN с «BERTopic из коробки», HDBSCAN без UMAP, KMeans и агломеративной кластеризацией
(очищенный текст, одна модель).
"""), code(r'''
s = pd.read_csv(RES / "sensitivity.csv") if (RES / "sensitivity.csv").exists() else pd.DataFrame()
s.set_index(["dataset", "config"])[["n_clusters", "outliers", "nmi", "bcubed_p", "bcubed_r", "purity", "clean_clusters"]].round(3) if len(s) else "нет данных"
'''), md("""
## 6. Можно ли дать кластеру понятное описание
Для каждого значимого кластера (≥ 1% текстов): частые слова (c-TF-IDF, как в BERTopic) и описание LLM по 20 текстам
(10 ближайших к центру + 10 случайных) — локальная Qwen3-4B. «Выбивающиеся» — тексты, которые сама модель
сочла не подходящими под описание.

**6.1. Наборы с метками** (geo — рубрика места, tickets — скрытая тема): название LLM рядом с главной меткой кластера и её
долей (чистота). Ниже — проверка самооценки LLM: совпадают ли её «выбивающиеся» с реально чужими по меткам текстами.
"""), code(r'''
for ds in ["geo", "tickets"]:
    p = RES / f"clusters_{ds}.json"
    if p.exists():
        cl = pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))["clusters"])
        display(cl[["id", "size", "name", "top_label", "top_label_share", "description"]].rename(columns={"top_label_share": "чистота"}))
p = RES / "descriptions_eval.csv"
if p.exists():
    ev = pd.read_csv(p)
    g = ev.groupby("dataset").agg(кластеров=("cluster", "size"), чужих_в_выборках=("alien", "sum"), отметила_LLM=("flagged", "sum"), из_них_чужих=("hit", "sum"))
    g["точность отметок"] = (g["из_них_чужих"] / g["отметила_LLM"]).round(2)
    g["полнота отметок"] = (g["из_них_чужих"] / g["чужих_в_выборках"]).round(2)
    display(g)
'''), md("""
**6.2. Отзывы о банках** (очищенный текст, min_cluster_size = 20; меток нет). Ниже — описания LLM и ручная проверка:
по 10 случайных текстов каждого кластера, оценка 1 / 0.5 / 0 (подходит / тема не главная / не подходит), оценка названия
и название, которое точнее отражает кластер. Проверял разработчик, а не аналитик заказчика.
"""), code(r'''
p = RES / "clusters_bank.json"
if p.exists():
    cl = pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))["clusters"])
    cols = [c for c in ["id", "size", "share", "name", "description", "sentiment", "homogeneity", "llm_outlier_share", "rating_mean", "neg_share", "keywords"] if c in cl.columns]
    display(cl[cols])
else:
    print("нет research/results/clusters_bank.json")
'''), code(r'''
p = RES / "manual_review_bank_scored.csv"
if p.exists():
    mr = pd.read_csv(p)
    w = (mr["relevant"] / mr["checked"] * mr["size"]).sum() / mr["size"].sum()
    print(f"Ручная проверка: {len(mr)} кластеров × 10 случайных текстов; доля подходящих под описание: "
          f"{mr['relevant'].sum() / mr['checked'].sum():.2f} (с учётом размера кластеров {w:.2f}; строго {mr['strict'].sum() / mr['checked'].sum():.2f})")
    print("Названия:", mr["name_ok"].value_counts().to_dict())
    display(mr[["cluster", "size", "name", "relevant", "checked", "name_ok", "better_name", "comment"]])
'''), md("""
## 7. Масштаб: 40 тысяч тикетов
Скорость моделей на 4 ядрах CPU (без GPU), замер в одиночку (`research/comments/speed.py`); оценка времени на 40 000 тикетов.\nНиже — прогон 40 000 синтетических тикетов целиком: очистка, эмбеддинги, UMAP + HDBSCAN, качество против скрытых тем.
"""), code(r'''
sp = pd.read_csv(RES / "speed_cpu.csv").rename(columns={"reviews_per_s": "отзывов/с (~960 симв.)", "tickets_per_s": "тикетов/с (~80 симв.)",
                                                       "min_40k_tickets": "40 тыс. тикетов, мин", "load_s": "загрузка, с"})
sp.set_index("model")
'''), code(r'''
p = RES / "scale_40k.csv"
pd.read_csv(p).set_index("model") if p.exists() else "нет данных"
'''), md("""
## 8. Проверка выводов на устойчивость (аудит)
**8.1. Что именно портит эмбеддинги и обобщается ли очистка** (`research/comments/robustness.py`, отзывы Яндекс Карт):
* `dirty` — основной замусоренный вариант (60% текстов: ссылка, имя, телефон; 40% — формат SMM), `clean` — после очистки;
* `tzdirty` / `tzclean` — **только** просьба ТЗ: в каждом тексте ссылка, имя и телефон, без SMM-формата;
* `altdirty` — те же виды мусора **в форматах, под которые правила не писались** (ссылки без https, телефоны со скобками,
  «Фамилия И. О.», ники, e-mail с точкой) и 40% текстов в виде выгрузки CRM («Обращение №…», «[14:02] Пользователь:»,
  «Сотрудник поддержки (Анна):», «--- Резюме ---», «Статус: закрыто»); `altclean` — правила после исправлений аудита
  (замер правил «как было», до исправлений, — вторая таблица); `altclean_bp` — перед правилами удаление шаблонов по
  частоте в корпусе (`lurker.clean.remove_boilerplate`);
* `clean_bp` — основной вариант с удалением шаблонов (проверка, что оно не вредит).

Сохранность — доля слов исходного отзыва в очищенном; «чужие слова» — доля слов очищенного текста, которых не было в
исходном (остатки обвязки); left_* — доля текстов, где остался кусок подмешанного телефона, ссылки, e-mail, имени;
fmt_nn_share — у текстов в «особом» формате (SMM или CRM) доля соседей того же формата (в корпусе таких 40%).
"""), code(r'''
p = RES / "robust_dirt.csv"
if p.exists():
    rd = pd.read_csv(p)
    cols = [c for c in ["nmi", "bcubed_p", "bcubed_r", "knn_overlap", "ari_vs_orig", "fmt_nn_share", "retention", "junk_share",
                        "left_phone", "left_link", "left_email", "left_name"] if c in rd.columns]
    order = ["orig", "dirty", "clean", "clean_bp", "tzdirty", "tzclean", "altdirty", "altclean", "altclean_bp"]
    rd["variant"] = pd.Categorical(rd["variant"], order)
    display(rd[rd.dataset == "geo"].sort_values(["model", "variant"]).set_index(["model", "variant"])[cols].round(3))
    display(rd[rd.dataset == "tickets"].set_index(["model", "variant"])[["nmi", "bcubed_p", "bcubed_r"]].round(3))
p = RES / "robust_dirt_before_fixes.csv"
if p.exists():
    print("Незнакомый мусор, правила и удаление шаблонов ДО исправлений аудита:")
    bf = pd.read_csv(p)
    display(bf[(bf.dataset == "geo") & bf.variant.isin(["altclean", "altclean_bp"])].set_index(["model", "variant"])
            [["nmi", "bcubed_p", "fmt_nn_share", "junk_share", "left_name", "left_phone"]].round(3))
'''), md("""
**8.2. Разброс от случайного зерна UMAP** (5 зёрен, те же эмбеддинги): различие моделей меньше разброса — шум.
"""), code(r'''
p = RES / "seeds.csv"
if p.exists():
    sd = pd.read_csv(p)
    display(sd.groupby(["dataset", "variant", "model"])[["nmi", "bcubed_p", "bcubed_r", "n_clusters"]].agg(["mean", "std"]).round(3))
'''), md("""
**8.3. Отзывы о банках: бренды и тональность кластеров.** В данных есть название банка и рейтинг — по ним видно, что
кластеры делятся сначала по тональности (благодарности отдельно от жалоб), а часть кластеров — по бренду.
"""), code(r'''
p = RES / "bank_clusters_brand_rating.csv"
if p.exists():
    display(pd.read_csv(p))
p = RES / "names_check.csv"
if p.exists():
    nc = pd.read_csv(p)
    print("Названия LLM против главных меток (geo, tickets):", nc.name_ok.value_counts().to_dict())
    display(nc[nc.name_ok != "да"])
'''), md("""
## 9. Динамика: «стали ли чаще жаловаться на домофон в приложении?»
С этого вопроса началось ТЗ. В синтетические тикеты при генерации заложены события по датам: всплеск жалоб на домофон в
приложении в марте, рост ошибок оплаты с июня, плавный рост расторжений, зимний сезон низкой скорости, рост угроз и
новая тема в августе. Кластеры (очищенный текст, USER-bge-m3, без разметки) считаются по неделям, поиск изменений —
`lurker.dynamics.detect_changes` (всплеск, новая тема, рост, спад); то же для сравнения — по настоящим темам
(`research/comments/dynamics_check.py`).
"""), code(r'''
p = RES / "dynamics_check.csv"
if p.exists():
    dc = pd.read_csv(p)
    print(f"заложенных событий найдено по кластерам: {dc['найдено по кластерам'].sum()} из {len(dc)}; "
          f"по настоящим темам: {dc['найдено по настоящим темам'].sum()} из {len(dc)}")
    display(dc.drop(columns=["находка"]))
    fd = pd.read_csv(RES / "dynamics_findings.csv")
    display(fd[fd.source == "кластеры"][["kind", "name", "topic", "period", "count", "baseline", "factor"]])
'''), code(r'''
from lurker.dynamics import timeline
t = pd.read_parquet(ROOT / "data" / "research" / "comments" / "tickets.parquet") if (ROOT / "data" / "research" / "comments" / "tickets.parquet").exists() else None
lab = ROOT / "build" / "research" / "clusters" / "tickets__clean__user-bge-m3.npy"
if t is not None and lab.exists():
    t["date"] = pd.to_datetime(t["date"]); t["cluster"] = np.load(lab)
    topic = "домофон_приложение_не_приходит_звонок"
    cls = t[t.label == topic].cluster.value_counts().index[:2].tolist()
    weeks = timeline(t, topic_col="label", freq="W")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(weeks.index, weeks[topic], color="#9aa0a6", lw=2, label="настоящая тема (разметка)")
    by_cl = timeline(t[t.cluster.isin(cls)].assign(g="кластеры"), topic_col="g", freq="W")["кластеры"].reindex(weeks.index, fill_value=0)
    ax.plot(weeks.index, by_cl, color="#1a73e8", lw=2, label="кластеры «Проблема с приложением домофона» (без разметки)")
    print(f"корреляция недельных рядов: {np.corrcoef(weeks[topic], by_cl)[0, 1]:.2f}")
    ax.set_title("Жалобы «домофон не звонит в приложение», тикетов в неделю"); ax.grid(axis="y", alpha=.3)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="upper right"); plt.tight_layout(); plt.show()
else:
    print("нужны data/research и build/research (запустить prepare_data.py и experiment.py)")
'''), md("""
## 10. Какие темы появляются и какие необычные проблемы встречаются
Два вопроса заказчика из ТЗ, которые кластеры сами по себе не закрывают (`research/comments/novelty.py`).

**10.1. Новые темы по месяцам.** Для каждого месяца: эталон — все тикеты до него; тикет «новый», если его 5 ближайших
соседей в эталоне дальше, чем у 99% тикетов эталона между собой; новые тикеты группируются, группа ≥ 5, у которой и центр
далёк от прошлого, — кандидат в новую тему (`lurker.novelty.emerging_groups`). В тикеты при генерации заложена одна новая тема (с 10 августа); остальные месяцы показывают ложные тревоги.
"""), code(r'''
p = RES / "novelty_emerging.csv"
pd.read_csv(p).drop(columns=["примеры"], errors="ignore") if p.exists() else "нет данных"
'''), md("""
**10.2. Необычные комментарии.** В очищенные тикеты подмешаны 50 написанных вручную необычных, но правдоподобных
обращений («роутер искрит», «домофон сам открывает дверь ночью», «в квитанции чужая фамилия») и — контроль — 50 написанных
тем же автором ОБЫЧНЫХ. Метод из задачи 2: новизна относительно 10 ближайших соседей — внутри кластера и по всему
корпусу; варианты «от 4 слов» не ранжируют короткие тексты (опечатки, «см. предыдущий»). Если контрольные тоже всплывают,
метод ловит стиль автора, а не необычный смысл. Ниже — та же новизна на настоящих отзывах о банках, топ-20 прочитан вручную.
"""), code(r'''
p = RES / "novelty_unusual.csv"
if p.exists():
    display(pd.read_csv(p)[["метод", "позитив", "n_pos", "доля", "ap", "auc", "p@20", "p@50", "помечено z≥2.5, %", "доля позитива среди z≥2.5"]])
    top = pd.read_csv(RES / "novelty_unusual_top.csv")
    print("Топ-40 самых необычных (по всему корпусу, тексты от 4 слов):", top["kind"].value_counts().to_dict())
    display(top.head(25))
p = RES / "novelty_bank_top20_manual.csv"
if p.exists():
    bm = pd.read_csv(p)
    print("Настоящие отзывы о банках, топ-20 самых необычных — ручная проверка:", bm["необычная проблема"].value_counts().to_dict())
    display(bm[["rank", "z", "rating", "необычная проблема", "комментарий", "text"]])
'''), md("""
## 11. Что положительного или отрицательного пишут: тональность без оценок
Готовые открытые модели тональности, без обучения (`research/comments/sentiment.py`). Проверка: отзывы о банках и
Яндекс Картах — по рейтингу (1–2 негатив, 3 нейтрально, 4–5 позитив), тикеты — по скрытой разметке генератора.
Тикеты — заметки оператора о клиенте, а не отзывы, поэтому для них проверка отдельная. Ниже — доля негатива по
кластерам: предсказанная моделью против настоящей.
"""), code(r'''
p = RES / "sentiment_eval.csv"
if p.exists():
    display(pd.read_csv(p).pivot_table(index="model", columns="dataset", values=["macro_f1", "auc_neg"]).round(3))
    display(pd.read_csv(p))
for ds in ("tickets", "bank"):
    p = RES / f"sentiment_clusters_{ds}.csv"
    if p.exists():
        ct = pd.read_csv(p)
        big = ct[ct["size"] >= 30]
        print(f"{ds}: доля негатива по кластерам — корреляция с настоящей {np.corrcoef(big.negative_true, big.negative_pred)[0, 1]:.2f}, "
              f"средняя ошибка {100 * (big.negative_true - big.negative_pred).abs().mean():.0f} п.п.")
        display(ct.head(20))
'''), md("""
## 12. Отзывы о банках: кластеры по брендам и «мешок» благодарностей
`research/comments/bank_brands.py`: (1) названия банков и их продуктов в тексте заменены словом «банк» — меньше ли
кластеров «по бренду»; (2) крупные кластеры (> 15% корпуса) кластеризованы второй раз внутри себя — подтемы.
"""), code(r'''
for f in ("bank_brands.csv", "bank_subclusters.csv"):
    p = RES / f
    if p.exists():
        display(pd.read_csv(p))
'''), md("""
## 13. 40 тысяч тикетов: настройки под объём
Рекомендованный конвейер целиком (удаление шаблонов → очистка → USER-bge-m3) на 40 000 синтетических тикетов и сетка
настроек: min_cluster_size и слияние близких кластеров по центроидам (`research/comments/scale_settings.py`).
"""), code(r'''
p = RES / "scale_40k_settings.csv"
pd.read_csv(p)[["min_cluster_size", "merge_threshold", "n_clusters", "nmi", "bcubed_p", "bcubed_r", "purity", "clean_clusters",
                "clean_s", "embed_s", "cluster_s"]] if p.exists() else "нет данных"
'''), md(conclusions("comments"))]
    return cells


def nb_calls():
    cells = [md("""
# Задача 2. RnD: аномальные звонки

**Гипотеза (ТЗ v2):** с помощью эмбеддингов и кластеризации можно автоматически находить звонки, которые сильно отличаются
от типичных звонков на ту же тему, даже если в них нет ключевых слов вроде мата.

**Схема:** кластеризуем все звонки по темам (UMAP → HDBSCAN), внутри темы считаем, насколько звонок выбивается,
выводим топ-N, проверяем по спискам искусственно испорченных и аномальных звонков (precision@K).

**Как воспроизвести:**
```
python3 research/calls/prepare_calls.py     # диалоги + испорченные + аномальные, списки id -> research/results/calls_lists.json
python3 research/calls/run_anomaly.py       # эмбеддинги, темы, оценки, метрики -> research/results/calls_metrics.csv
python3 research/calls/extra.py             # доп. проверки -> calls_extra.csv, calls_splice_sanity.csv
python3 research/make_notebooks.py
```
"""), code(SETUP), md("""
## 1. Данные
Датасет из ТЗ `katarinaaaaa/evaluation-of-customer-service`: 2502 уникальных диалога «оператор — клиент» (2497 — с 4+ репликами) из десятков
бизнесов (доставка, фитнес, банки, связь, IT-продажи), у каждого тема и оценки работы оператора 0–10.

**База — 2036 вежливых диалогов** (вежливость оператора ≥ 6). В датасете у 18% диалогов вежливость оператора ≤ 5 из 10, у 6% — ≤ 2 («Чё хотел?»), и тогда
грубость — не аномалия; в ТЗ посылка обратная: «операторы у нас не хамят, как правило». В базу подмешано:

* **испорченные** (163, 8%) — артефакты распознавания речи: выпавшие гласные («домофон» → «дмфон»), неуместные слова
  («кошка», «кактус»), слипшиеся слова. **Аномалиями не являются** — метод не должен их поднимать;
* **аномальные** (122, 6%): `splice` — середина диалога (40% реплик) подменена куском диалога из далёкой области
  (ТЗ: «банковский диалог в звонке про ремонт»); `splice_near` — то же из другой темы, но ближе по словам; `rude` — одна
  реплика оператора заменена грубостью; `absurd` — оператор при нормальном тоне несёт бред или нарушает бизнес-логику
  (пример из ТЗ «ремонт платный и у абонента сифилис», перевод на личную карту, просьба назвать CVC-код).
  Грубых и абсурдных реплик по 60 уникальных: одинаковые вставки находили бы друг друга как соседей;
* **естественно невежливые** (51, 2.5%) — нетронутые диалоги с оценкой вежливости ≤ 2 по разметке датасета.

Все тексты приведены к виду расшифровки (нижний регистр, без пунктуации): так выглядят реальные звонки, и испорченные
не выделяются одним только стилем. Списки id — `research/results/calls_lists.json`.
"""), code(r"""
calls = pd.read_parquet(DATA / "calls" / "calls.parquet")
display(calls.groupby(["kind", "anomaly_type"]).size().to_frame("звонков"))
for ty in ("splice", "rude", "absurd"):
    r = calls[calls.anomaly_type == ty].iloc[1]
    print(f"===== {ty}: {r['theme']}\n{r['text'][:800]}\n")
r = calls[calls.kind == "corrupted"].iloc[1]
print(f"===== испорченный: {r['theme']}\n{r['text'][:500]}")
"""), md("""
## 2. Насколько хорошо находятся аномалии
Методы (все оценки — робастный z внутри темы):
* **centroid** — расстояние звонка до центра темы (схема из ТЗ); **knn** — до 10 ближайших звонков темы; **lof** — Local Outlier Factor;
* **turns** — по репликам: каждая реплика оператора сравнивается с 10 ближайшими репликами операторов в *других* звонках той
  же темы, звонок оценивается по самой непохожей реплике; **turns_c** — то же на центрированных эмбеддингах реплик
  (вычтен средний вектор: у однотипных реплик он забивает смысловые различия);
* **coherence** — связность частей звонка; **relevance** — ответ оператора против предыдущей реплики клиента; комбинации.

Метрики: AP (average precision; случайный уровень = доля аномалий, 0.06), AUC, P@K — доля настоящих аномалий среди K самых
подозрительных (аномалий 122), «шум@K» — доля испорченных (ложные тревоги от шума STT) в топ-K.
"""), code(r"""
cm = pd.read_csv(RES / "calls_metrics.csv")
cols = ["model", "method", "n_topics", "ap", "auc", "p@40", "p@120", "r@120", "noise@120", "ap_splice", "ap_splice_near", "ap_rude", "ap_absurd", "ap_natural"]
main = cm[cm.method.isin(["centroid", "knn", "lof", "turns", "turns_c", "coherence", "relevance", "turns+coherence"])]
display(main[cols].sort_values(["method", "ap"], ascending=[True, False]).round(3).reset_index(drop=True))
"""), code(r"""
order = [m for m in ["tfidf", "rubert-tiny2", "e5-small", "user-base", "user-bge-m3"] if m in set(cm.model)]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
t = cm[cm.method.isin(["centroid", "turns_c"])].pivot(index="model", columns="method", values="ap").reindex(order)
w = 0.38
for j, (meth, col, lab) in enumerate((("centroid", "#2a78d6", "звонок целиком: расстояние до центра темы (ТЗ)"), ("turns_c", "#eb6834", "по репликам оператора внутри темы (центрированные)"))):
    axes[0].bar(np.arange(len(order)) + (j - 0.5) * w, t[meth].values, width=w * 0.92, color=col, label=lab, zorder=3)
axes[0].axhline(0.06, color="#52514e", lw=1, ls="--"); axes[0].text(len(order) - 0.5, 0.065, "случайно", ha="right", fontsize=9, color="#52514e")
axes[0].set_xticks(np.arange(len(order)), order, rotation=15); axes[0].set_ylim(0, 0.5); axes[0].set_ylabel("average precision")
axes[0].set_title("Все подмешанные аномалии", loc="left"); axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.2))
b = cm[cm.method == "turns_c"].set_index("model").reindex(order)
types = [("ap_rude", "грубость", "#2a78d6"), ("ap_absurd", "бред / нарушение логики", "#eb6834"), ("ap_splice", "подмена куска", "#1baf7a"), ("ap_natural", "естественные хамы", "#eda100")]
w = 0.8 / len(types)
for j, (c, lab, col) in enumerate(types):
    axes[1].bar(np.arange(len(order)) + (j - 1.5) * w, b[c].values, width=w * 0.92, color=col, label=lab, zorder=3)
axes[1].set_xticks(np.arange(len(order)), order, rotation=15); axes[1].set_ylim(0, 0.5); axes[1].set_ylabel("average precision")
axes[1].set_title("По типам, метод turns_c", loc="left"); axes[1].legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.2))
plt.tight_layout(); plt.show()
"""), md("""
## 3. Почему звонок целиком не работает, и нужны ли темы
Дополнительные проверки (`research/calls/extra.py`): сходство подменённого звонка с его же исходной версией против сходства
двух обычных звонков одной темы; оценка по репликам внутри темы против всего корпуса без тем; центрирование эмбеддингов.
"""), code(r"""
p1, p2 = RES / "calls_splice_sanity.csv", RES / "calls_extra.csv"
if p1.exists():
    display(pd.read_csv(p1))
if p2.exists():
    ex = pd.read_csv(p2)
    display(ex[["model", "mean_pair_cos", "check", "ap", "auc", "p@120", "noise@120", "ap_splice", "ap_rude", "ap_absurd", "ap_natural"]].round(3))
"""), md("""
## 4. Порог и топ-N для контроля качества
В день ~3000 звонков. Практичный режим — ежедневный топ-N самых подозрительных на ручную проверку контролёру. Ниже —
точность и полнота на наших данных (доля аномалий 6%) и пересчёт точности на реалистичную долю аномалий (1% и 0.5%)
при тех же долях срабатываний на аномальных и нормальных звонках.
"""), code(r"""
bm = cm[cm.method == "turns_c"].sort_values("auc", ascending=False).iloc[0]
p = ROOT / "build" / "research" / f"calls_scores__stt__{bm['model']}.parquet"
p = p if p.exists() else RES / f"calls_scores_{bm['model']}.parquet"          # копия в git: ноутбук перезапускается из чистого клона
sc = pd.read_parquet(p).merge(calls[["id", "kind", "anomaly_type"]], on="id")
pool = sc.kind != "natural"
s = sc.loc[pool, "turns_c"].to_numpy(); an = (sc.kind[pool] == "anomaly").to_numpy(); co = (sc.kind[pool] == "corrupted").to_numpy()
rows = []
for thr in (1.5, 2, 2.5, 3):
    f = s >= thr
    tpr, fpr = (s[an] >= thr).mean(), (s[~an] >= thr).mean()
    row = {"порог z": thr, "помечено, %": 100 * f.mean(), "точность": an[f].mean() if f.any() else 0, "полнота": tpr, "шум среди помеченных": co[f].mean() if f.any() else 0}
    for prev in (0.01, 0.005):
        row[f"точность при доле {prev:.1%}"] = prev * tpr / (prev * tpr + (1 - prev) * fpr) if tpr + fpr else 0
        row[f"помечено в день из 3000 при {prev:.1%}"] = round(3000 * (prev * tpr + (1 - prev) * fpr))
    rows.append(row)
print(f"конфигурация: {bm['model']} / turns_c (центрированные реплики оператора внутри темы)")
display(pd.DataFrame(rows).round(3))
o = np.argsort(-s)
pd.DataFrame([{"топ-N": n, "настоящих аномалий": int(an[o[:n]].sum()), "точность": an[o[:n]].mean(), "испорченных (шум)": int(co[o[:n]].sum())} for n in (20, 40, 60, 100, 150)]).round(2)
"""), code(r"""
top = sc.assign(score=sc["turns_c"]).sort_values("score", ascending=False).head(15).merge(calls[["id", "theme", "turns"]], on="id")
for _, r in top.iterrows():
    print(f"[z={r['score']:.1f}] {r['kind']:9} {r['anomaly_type'] or '':11} | {r['theme']}")
"""), md("""
## 5. Проверка выводов (аудит)
**5.1. Доверительные интервалы** (`research/calls/bootstrap.py`, бутстреп по звонкам, 1000 повторов, 95%):
"""), code(r"""
p = RES / "calls_bootstrap.csv"
if p.exists():
    bs = pd.read_csv(p)
    bs["значение [95% ДИ]"] = bs.apply(lambda r: f"{r.value:.2f} [{r.ci_low:.2f}–{r.ci_high:.2f}]", axis=1)
    display(bs.pivot_table(index=["model", "method"], columns="metric", values="значение [95% ДИ]", aggfunc="first")
            [["ap", "auc", "p@40", "p@120", "precision_z2", "recall_z2", "flagged"]])
"""), md("""
**5.2. Ручной разбор топ-40** (ТЗ: «как проверить, что аномалия является таковой — вручную»). 40 самых подозрительных
звонков прочитаны вслепую, без разметки (читал разработчик): «аномалия» — хамство, абсурд, чужой кусок диалога;
«пограничный» — звонок из исходного датасета, который контролёру стоит посмотреть (некомпетентность, скрытые доплаты,
нарушение безопасности); потом сверено с разметкой.
"""), code(r"""
p = RES / "calls_top40_manual.csv"
if p.exists():
    mt = pd.read_csv(p)
    display(pd.crosstab(mt["blind"], mt["kind"], margins=True))
    display(mt[mt.blind == "пограничный"][["rank", "id", "note", "politeness", "regulation"]])
"""), md("""
**5.3. Находит ли оценка естественно слабые звонки.** Среди нетронутых звонков — связь z с оценками работы оператора
из разметки самого датасета (0–10): AUC «оценка ≤ 6», ранговая корреляция, доля слабых в топ-40 и топ-100 нормальных
звонков против доли во всех нормальных.
"""), code(r"""
p = RES / "calls_grades.csv"
pd.read_csv(p) if p.exists() else "нет данных"
"""), md("""
**5.4. Вне выборки: Russian MultiWOZ** (`research/calls/multiwoz_check.py`). Другой корпус (письменные диалоги о
бронировании гостиниц, ресторанов, поездов, такси), 3000 диалогов — дневной объём заказчика. Порча и аномалии теми же
функциями и в тех же долях; `splice` — кусок банковского диалога посреди диалога о бронировании (пример из ТЗ).
Скрипт `research/run_calls.py` запущен как есть: модель, метод и порог выбраны на основном корпусе, здесь ничего не
подбиралось.
"""), code(r"""
p = RES / "mwoz_metrics.csv"
pd.read_csv(p) if p.exists() else "нет данных"
"""), md("""
**5.5. Подмена куска разговора — отдельный сигнал.** Подмену основная оценка ловит хуже всего. Проверены две добавки:
новизна реплик **клиента** (в подмешанном куске чужие и они) и «**чужая тема по репликам**» (`lurker.anomaly.offtopic_score`:
реплика ближе к центру другой темы, чем к своей). Обе ловят подмену лучше основной оценки, но, смешанные с ней, теряют на
грубости и бреде — поэтому их стоит вести отдельным списком, а не одной оценкой.
"""), code(r"""
cols = ["model", "method", "ap", "auc", "p@40", "p@120", "noise@120", "ap_splice", "ap_splice_near", "ap_rude", "ap_absurd"]
cm[(cm.variant == "stt") & cm.model.isin(["user-base", "tfidf"]) & cm.method.isin(["turns_c", "client_c", "offtopic", "op+client_m1", "op+offtopic"])][cols].round(3)
"""), md(conclusions("calls"))]
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    a = ap.parse_args()
    for name, cells in (("01_comments_clustering", nb_comments()), ("02_anomalous_calls", nb_calls())):
        nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                        "language_info": {"name": "python"}})
        path = HERE / f"{name}.ipynb"
        if not a.no_exec:
            from nbconvert.preprocessors import ExecutePreprocessor
            ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
        nbf.write(nb, path)
        print(f"→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
