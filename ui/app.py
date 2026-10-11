"""Streamlit-интерфейс, сценарий «сначала вопрос» (ТЗ п. 2, 8): загрузка файла → произвольный промпт →
динамика понятия из промпта, тематики внутри выборки, ассистент по выборке. Обзор всех тем — отдельная вкладка.
Запуск: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lurker.service.assistant import answer  # noqa: E402
from lurker.config import load_config  # noqa: E402
from lurker.data import STANDARD, ColumnMap, guess_columns, load_table, normalize  # noqa: E402
from lurker.dynamics import detect_changes, timeline  # noqa: E402
from lurker.llm import LLMClient  # noqa: E402
from lurker.period import describe  # noqa: E402
from lurker.pipeline import run_analysis  # noqa: E402
from lurker.preclass import parse_criteria  # noqa: E402
from lurker.service.query import borderline, run_query  # noqa: E402
from lurker.summarize import topics_table  # noqa: E402

# Категориальная палитра: фиксированный порядок слотов, цвет закрепляется за серией и не меняется при фильтрах.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#9a9993"
FREQ_LABELS = {"день": "D", "неделя": "W", "месяц": "M"}
ROLE_LABELS = {"text": "Комментарий *", "date": "Дата *", "id": "ID", "service": "Сервис", "subservice": "Подкатегория",
               "author": "Сотрудник", "department": "Отдел"}
EXAMPLES = ["как часто нас оскорбляли и угрожали физической расправой за последние 2 месяца",
            "стали ли чаще жаловаться на домофон в приложении?",
            "сравни, что говорили про мобильное приложение в январе и в июле",
            "как изменилась динамика упоминания домофона за последние полгода",
            "что изменилось за последний месяц"]

st.set_page_config(page_title="Review-lurker", page_icon="🔎", layout="wide")
cfg = load_config()


# ---------- кэшируемые шаги ----------

@st.cache_data(show_spinner=False)
def read_upload(name: str, data: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as f:
        f.write(data)
        path = f.name
    return load_table(path)


@st.cache_resource(show_spinner=False)
def analyze(key: str, df: pd.DataFrame, freq: str, backend: str, method: str, max_topics: int | None, use_llm: bool):
    llm = LLMClient(cfg.llm, cache_dir=cfg.cache.dir) if use_llm else None
    return run_analysis(df, cfg, llm=llm, freq=freq, backend=backend, method=method, max_topics=max_topics,
                        cache_dir=cfg.cache.dir)


@st.cache_resource(show_spinner=False)
def query(key: str, prompt: str, threshold: float, freq: str, use_llm: bool, criteria_text: str):
    an = st.session_state["analysis"]
    llm = LLMClient(cfg.llm, cache_dir=cfg.cache.dir) if use_llm else None
    crit = parse_criteria(criteria_text) if criteria_text.strip() else None
    return run_query(an, prompt, threshold=threshold, freq=freq, llm=llm, criteria=crit)


def color_for(name: str) -> str:
    cmap = st.session_state.setdefault("color_map", {})
    if name not in cmap:
        used = set(cmap.values())
        free = [c for c in PALETTE if c not in used]
        cmap[name] = free[0] if free else GRAY
    return cmap[name]


def line_chart(tl: pd.DataFrame, ytitle: str, shares: bool = False, height: int = 380) -> go.Figure:
    fig = go.Figure()
    for col in tl.columns:
        fig.add_trace(go.Scatter(x=tl.index, y=tl[col], name=str(col), mode="lines+markers",
                                 line=dict(width=2, color=color_for(str(col))), marker=dict(size=6),
                                 hovertemplate="%{y:.1%}" if shares else "%{y}"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(title=ytitle, gridcolor="rgba(128,128,128,0.2)",
                                                                  tickformat=".0%" if shares else None, rangemode="tozero"),
                      xaxis=dict(title=None, gridcolor="rgba(0,0,0,0)"))
    return fig


def bar_chart(s: pd.Series, ytitle: str, color: str = PALETTE[0], height: int = 300) -> go.Figure:
    fig = go.Figure(go.Bar(x=[str(i) for i in s.index], y=s.values, marker=dict(color=color), hovertemplate="%{y}"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(title=ytitle, gridcolor="rgba(128,128,128,0.2)"), bargap=0.25)
    return fig


def period_index(idx: pd.DatetimeIndex, freq: str) -> list[str]:
    fmt = {"D": "%d.%m", "W": "%d.%m", "M": "%m.%Y"}.get(freq, "%d.%m.%Y")
    return [d.strftime(fmt) for d in idx]


# ---------- боковая панель ----------

st.sidebar.header("Данные")
upload = st.sidebar.file_uploader("CSV / XLSX (до 200 МБ)", type=["csv", "xlsx", "xls"])
use_test = st.sidebar.checkbox("Тестовый набор data/test/tickets.csv", value=upload is None)

raw = None
source_key = ""
if upload is not None:
    data = upload.getvalue()
    raw = read_upload(upload.name, data)
    source_key = hashlib.sha1(data).hexdigest()[:12]
elif use_test:
    p = ROOT / "data/test/tickets.csv"
    if p.is_file():
        raw = load_table(p)
        source_key = "test:" + hashlib.sha1(p.read_bytes()).hexdigest()[:12]

cmap = None
if raw is not None:
    st.sidebar.subheader("Колонки")
    try:
        guessed = guess_columns(raw).as_dict()
    except ValueError:
        guessed = {r: None for r in STANDARD}
    options = ["—"] + list(raw.columns)
    chosen = {}
    for role in ("text", "date", "service", "subservice", "author", "department", "id"):
        default = guessed.get(role)
        idx = options.index(default) if default in options else 0
        chosen[role] = st.sidebar.selectbox(ROLE_LABELS[role], options, index=idx, key=f"col_{role}")
    if chosen["text"] == "—" or chosen["date"] == "—":
        st.sidebar.error("Нужны колонки с комментарием и датой.")
    else:
        cmap = ColumnMap(**{k: (None if v == "—" else v) for k, v in chosen.items()})

with st.sidebar.expander("Параметры", expanded=False):
    backend = st.selectbox("Эмбеддинги", ["hybrid", "tfidf", "navec", "st"], index=0,
                           help="hybrid/tfidf/navec работают офлайн; st — sentence-transformers, нужна модель из конфига")
    method = st.selectbox("Кластеризация (обзор тем)", ["umap_hdbscan", "bertopic", "agglomerative"], index=0)
    max_topics = st.number_input("Укрупнить обзорные темы до N (0 — нет)", min_value=0, max_value=200,
                                 value=int(cfg.cluster.max_topics or 0))
    llm_ok = bool(cfg.llm.api_key) or "localhost" in cfg.llm.base_url or "127.0.0.1" in cfg.llm.base_url
    use_llm = st.checkbox(f"LLM: {cfg.llm.model} @ {cfg.llm.base_url}", value=False, disabled=not llm_ok,
                          help="Ключ берётся из переменной окружения " + cfg.llm.api_key_env)
    if not llm_ok:
        st.caption("LLM выключен: нет ключа. Понятие из промпта — по смыслу через эмбеддинги и словарь, "
                   "названия тем — по ключевым словам, ответы — по счётчикам и цитатам.")

run = st.sidebar.button("Загрузить и подготовить", type="primary", disabled=cmap is None,
                        help="Очистка текстов, эмбеддинги, обзорные темы. Дальше — любые вопросы без пересчёта.")

# ---------- основная область ----------

st.title("🔎 Review-lurker")
st.caption("Семантический анализ комментариев к тикетам: вопрос → выборка по смыслу → динамика → тематики → ассистент")

if raw is None:
    st.info("Загрузи CSV/XLSX слева или включи тестовый набор.")
    st.stop()

if run and cmap is not None:
    st.session_state["analysis_args"] = (source_key + "|" + "|".join(str(v) for v in cmap.as_dict().values()),
                                         cmap, "W", backend, method, max_topics or None, use_llm)
    st.session_state.pop("query_result", None)

args = st.session_state.get("analysis_args")
if not args:
    st.write(f"Строк: **{len(raw)}**, колонок: {len(raw.columns)}. Проверь маппинг колонок слева и нажми «Загрузить и подготовить».")
    st.dataframe(raw.head(20), use_container_width=True)
    st.stop()

key, cmap_used, freq_base, backend_used, method_used, max_topics_used, use_llm_used = args
akey = key + f"|{freq_base}|{backend_used}|{method_used}|{max_topics_used}|{use_llm_used}"
with st.spinner("Очистка, эмбеддинги, обзорные темы…"):
    an = analyze(akey, normalize(raw, cmap_used), freq_base, backend_used, method_used, max_topics_used, use_llm_used)
st.session_state["analysis"] = an
llm = LLMClient(cfg.llm, cache_dir=cfg.cache.dir) if use_llm_used else None
df = an.df

c1, c2, c3, c4 = st.columns(4)
c1.metric("Обращений", len(df))
c2.metric("Период данных", f"{df['date'].min():%d.%m.%y} – {df['date'].max():%d.%m.%y}")
c3.metric("Отделов / сервисов", f"{df['department'].nunique()} / {df['service'].nunique()}")
c4.metric("Обзорных тем", len(an.clusters.topics))

# ================= ВОПРОС =================
st.subheader("Вопрос к данным")
qcol, pcol = st.columns([4, 1])
prompt = qcol.text_input("Что посмотреть? Период можно писать прямо в вопросе.",
                         placeholder=EXAMPLES[0], key="prompt")
freq_label = pcol.radio("Гранулярность", list(FREQ_LABELS), index=2, horizontal=True, key="freq_q")
with st.expander("Примеры вопросов и тонкая настройка", expanded=False):
    ex = st.selectbox("Примеры", ["—"] + EXAMPLES, key="example")
    if ex != "—" and ex != st.session_state.get("prompt"):
        st.session_state["prompt"] = ex
        st.rerun()
    threshold = st.slider("Порог отбора по смыслу (ниже — шире выборка)", 0.1, 0.9, float(cfg.preclass.threshold), 0.05, key="thr")
    criteria_text = st.text_area("Критерии вручную (необязательно): «метка: описание | примеры: …; … | слова: …», по строке на метку",
                                 value="", height=80, key="criteria_text",
                                 help="Если задано, понятия берутся отсюда, а не из вопроса. С LLM критерии генерируются из вопроса автоматически.")
threshold = st.session_state.get("thr", float(cfg.preclass.threshold))
criteria_text = st.session_state.get("criteria_text", "")

if prompt and prompt.strip():
    freq_q = FREQ_LABELS[freq_label]
    with st.spinner("Отбор по смыслу, динамика, тематики…"):
        q = query(akey, prompt.strip(), threshold, freq_q, use_llm_used, criteria_text)
    what = " / ".join(c.name for c in q.concepts) or "все обращения"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Понятие", what[:40])
    m2.metric("Период", describe(q.periods).split(" (")[0][:28])
    m3.metric("Релевантных обращений", f"{q.n_relevant} из {q.n_scope}")
    if q.compare is not None:
        a, b = q.compare.index[:2]
        m4.metric(f"Доля: {a} → {b}", f"{q.compare.loc[b, 'доля']:.1%}", delta=f"{(q.compare.loc[b, 'доля'] - q.compare.loc[a, 'доля']) * 100:+.1f} п.п.")
    else:
        m4.metric("Доля в периоде", f"{q.n_relevant / q.n_scope:.1%}" if q.n_scope else "—")
    if q.concepts and q.concepts[0].keywords:
        st.caption("Ищу по смыслу: " + ", ".join(q.concepts[0].keywords[:14]) + (" …" if len(q.concepts[0].keywords) > 14 else "")
                   + (f" · пресеты: {', '.join(q.concepts[0].presets)}" if q.concepts[0].presets else ""))

    # --- динамика понятия
    st.markdown("**Динамика**")
    mode = st.radio("Показывать", ["число обращений", "долю от всех обращений"], horizontal=True, key="mode_q", label_visibility="collapsed")
    series = q.series.copy()
    if len(q.concepts) > 1:
        cols = [f"count_{c.name}" for c in q.concepts]
        tl = series[cols].rename(columns=lambda c: c[6:])
        if mode != "число обращений":
            tl = tl.div(series["total"].replace(0, np.nan), axis=0).fillna(0)
    else:
        tl = series[["count" if mode == "число обращений" else "share"]].rename(columns={"count": what, "share": what})
    if len(tl):
        for p in q.periods:
            pass
        fig = line_chart(tl, "обращений" if mode == "число обращений" else "доля", shares=(mode != "число обращений"))
        for p in q.periods:  # подсветка запрошенных периодов
            fig.add_vrect(x0=pd.Timestamp(p.start), x1=pd.Timestamp(p.end) + pd.Timedelta(days=1), fillcolor="rgba(128,128,128,0.12)", line_width=0,
                          annotation_text=p.label, annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)
    if q.findings:
        for f in q.findings[:4]:
            st.write("- " + f.describe(what))
    else:
        st.write("- заметных всплесков и трендов нет")
    if q.compare is not None:
        st.markdown("**Сравнение периодов**")
        st.dataframe(q.compare.style.format({"доля": "{:.1%}"}), use_container_width=True)

    # --- тематики внутри выборки
    st.markdown(f"**Тематики внутри выборки** ({q.n_relevant} обращений)")
    if q.subsummaries:
        table = topics_table(q.subsummaries)
        st.dataframe(table.style.format({"доля": "{:.1%}"}), use_container_width=True, hide_index=True)
        names = [s.name for s in q.subsummaries]
        pick = st.selectbox("Подробнее", names, key="subtopic_pick")
        s = next(x for x in q.subsummaries if x.name == pick)
        for d, t in s.quotes:
            st.markdown(f"- {d}: «{t}»")
    elif q.n_relevant:
        st.write("Обращений мало для разбивки на тематики; вот они:")
        st.dataframe(q.relevant(df)[["date", "department", "service", "clean"]].sort_values("date", ascending=False)
                     .rename(columns={"date": "дата", "department": "отдел", "service": "сервис", "clean": "текст"}),
                     use_container_width=True, hide_index=True)
    else:
        st.write("Ничего не найдено. Снизь порог отбора или переформулируй.")

    # --- ассистент по выборке
    st.markdown("**Уточняющий вопрос по этой выборке**")
    follow = st.text_input("Например: «что именно не работает?», «в каких отделах чаще?», «что говорили в августе?»", key="follow")
    if follow and q.n_relevant:
        sub = df[q.mask]
        ans = answer(follow, sub.reset_index(drop=True), an.E[q.mask], an.embedder, llm=llm, freq=q.freq, names=an.names)
        body = ans.text.split("\n", 1)[1] if ans.text.startswith("Период:") and "\n" in ans.text else ans.text
        st.markdown(body.replace("\n", "  \n"))
        with st.expander(f"Цитаты ({len(ans.quotes)})"):
            st.dataframe(pd.DataFrame(ans.quotes)[["date", "text"]].rename(columns={"date": "дата", "text": "текст"}),
                         use_container_width=True, hide_index=True)

    # --- калибровка
    with st.expander("Калибровка отбора: обращения у порога и выборка целиком"):
        st.caption("Если сюда попадает лишнее — подними порог; если релевантное осталось за бортом — опусти.")
        near = borderline(q, df)
        if len(near):
            st.dataframe(near.rename(columns={"date": "дата", "clean": "текст", "score": "оценка"}).style.format({"оценка": "{:.2f}"}),
                         use_container_width=True, hide_index=True)
        buf = io.StringIO()
        df[q.mask].drop(columns=["topic"], errors="ignore").to_csv(buf, index=False, sep=";")
        st.download_button("Скачать выборку (CSV)", buf.getvalue().encode("utf-8-sig"), file_name="selection.csv", mime="text/csv")

# ================= ОБЗОР =================
st.divider()
tab_over, tab_data = st.tabs(["Обзор всех тем", "Данные"])

with tab_over:
    st.caption("Автоматические темы по всему файлу (UMAP → HDBSCAN → c-TF-IDF) — чтобы увидеть, о чём вообще пишут, и что изменилось без подсказки.")
    fcols = st.columns(4)
    filters = {}
    for i, (col, label) in enumerate((("service", "Сервис"), ("department", "Отдел"), ("author", "Сотрудник"))):
        vals = sorted(v for v in df[col].unique() if v)
        if vals:
            sel = fcols[i].multiselect(label, vals, key=f"f_{col}")
            if sel:
                filters[col] = sel
    freq_o = fcols[3].radio("Гранулярность", list(FREQ_LABELS), index=1, horizontal=True, key="freq_o")
    tl_all = timeline(df, "topic_name", FREQ_LABELS[freq_o], filters)
    order = [an.topic_name(t.id) for t in an.clusters.topics] + ["прочее"]
    present = [t for t in order if t in tl_all.columns]
    picked = st.multiselect("Темы на графике (до 8)", present, default=[t for t in present if t != "прочее"][:6], max_selections=8, key="topics_o")
    if picked and not tl_all.empty:
        st.plotly_chart(line_chart(tl_all[picked], "обращений"), use_container_width=True)
    found = detect_changes(timeline(df, "topic", FREQ_LABELS[freq_o], filters))
    st.markdown("**Что изменилось**")
    for f in found[:12]:
        st.write("- " + f.describe(an.topic_name(f.topic)))
    if not found:
        st.write("Заметных изменений нет.")
    st.dataframe(topics_table(an.summaries).style.format({"доля": "{:.1%}"}), use_container_width=True, hide_index=True, height=400)

with tab_data:
    search = st.text_input("Поиск по тексту", key="search")
    view = df[["date", "department", "service", "author", "topic_name", "text", "clean"]]
    if search:
        view = view[view["text"].str.contains(search, case=False, regex=False) | view["clean"].str.contains(search, case=False, regex=False)]
    st.dataframe(view.rename(columns={"date": "дата", "department": "отдел", "service": "сервис", "author": "сотрудник",
                                      "topic_name": "тема", "text": "исходный текст", "clean": "очищенный"}),
                 use_container_width=True, hide_index=True, height=500)
    buf = io.StringIO()
    df.drop(columns=["topic"]).to_csv(buf, index=False, sep=";")
    st.download_button("Скачать с очищенным текстом и темами (CSV)", buf.getvalue().encode("utf-8-sig"),
                       file_name="tickets_clean.csv", mime="text/csv")
