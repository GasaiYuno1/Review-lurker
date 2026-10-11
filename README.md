# Review-lurker

Исследовательский прототип по ТЗ v2 (`docs/TZ.md`): две гипотезы про эмбеддинги и кластеризацию на русских текстах.

1. **Семантический анализ комментариев** — группируют ли эмбеддинги и кластеризация похожие комментарии без обучения,
   сколько в группах мусора, можно ли группы понятно описать и как всё это зависит от очистки текста от ссылок и ПД.
   Ноутбук `research/01_comments_clustering.ipynb`.
2. **Аномальные звонки** — находят ли эмбеддинги звонки, которые выбиваются из звонков на ту же тему, без ключевых слов.
   Ноутбук `research/02_anomalous_calls.ipynb`.

Итоговые выводы по обеим гипотезам — `research/REPORT.md`, коротко для заказчика — страница `research/report.html`. Первая версия ТЗ (сервис с интерфейсом) — `docs/TZ-v1-service.md`;
интерфейс от неё сохранён для визуальных проверок в отделяемой папке `ui/`.

## Быстрый старт
```
python3 -m pip install -r requirements.txt && python3 -m pip install --no-deps -r requirements-nodeps.txt && python3 -m pip install -e .
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu && python3 -m pip install -r requirements-research.txt
python3 -m unittest discover -s tests                     # тесты (без сети, ~1 мин)

# задача 1: данные (Kaggle, huggingface — без логина) → эмбеддинги → кластеры → описания → ноутбук
# (`python3 -m pip`, а не `pip`: в некоторых системах `pip` привязан к другой версии Python)
python3 research/comments/prepare_data.py
python3 research/comments/embed.py                         # долго на CPU: кэш в build/research/emb
python3 research/comments/experiment.py
python3 research/comments/sensitivity.py
python3 research/comments/describe.py --dataset bank --variant clean --model user-base   # нужен LLM, см. ниже
# задача 2
python3 research/calls/prepare_calls.py
python3 research/calls/run_anomaly.py
# ноутбуки с результатами
# аудит выводов: устойчивость очистки и зёрен, интервалы, проверка вне выборки (Russian MultiWOZ)
python3 research/comments/robustness.py && python3 research/calls/bootstrap.py
git clone https://github.com/vonCount/Russian-MultiWOZ /tmp/mwoz && python3 research/calls/multiwoz_check.py prepare /tmp/mwoz
python3 research/calls/multiwoz_check.py run && python3 research/calls/multiwoz_check.py eval
python3 research/make_notebooks.py
```

## На своих данных
```
python3 research/run_comments.py tickets.xlsx --text "Комментарий" --date "Дата" [--llm configs/llamacpp.yaml]   # USER-bge-m3; длинные тексты: --model user-base
python3 research/run_calls.py calls.csv --text "Расшифровка" --id "ID"      # или --long --id --speaker --text
```
Результаты — в `build/run_comments/` и `build/run_calls/` (таблицы кластеров, разметка строк, лист ручной проверки,
топ подозрительных звонков). Подробности и ожидаемое время — `research/REPORT.md`.

## Что внутри
- `src/lurker/` — ядро: `clean` (очистка правилами + NER natasha, удаление шаблонов; LLM few-shot поверх), `novelty`
  (новые темы, необычные комментарии), `sentiment` (тональность готовой моделью), `dirty` (замусоривание для
  экспериментов), `embed` (TF-IDF+SVD, navec, гибрид, нейросетевые модели: rubert-tiny2, multilingual-e5, USER-base,
  USER-bge-m3, FRIDA), `cluster` (UMAP → HDBSCAN → c-TF-IDF, BERTopic, KMeans/агломеративная; метрики NMI, ARI, B³,
  kNN-перекрытие), `anomaly` (аномальность внутри темы: центр, kNN, LOF, по репликам, связность, уместность ответа;
  precision@K), `summarize` (описание кластера LLM по 10–30 текстам), `llm` (OpenAI-совместимый клиент с кэшем),
  `text`, `data`, `config`, `dynamics`, `period`, `preclass`, `pipeline`, `cli`.
- `src/lurker/service/` и `ui/` — сценарий «вопрос к данным» и Streamlit-интерфейс из первой версии ТЗ; отделяемые
  (удалить папки и `tests/test_service.py` — ядро и исследование работают, проверяет `tests/test_core_standalone.py`).
- `research/` — скрипты экспериментов, ноутбуки, `results/` (небольшие итоговые таблицы в git), `REPORT.md`, `report.html`.
- `tools/` — генератор синтетических тикетов со скрытой разметкой, старые замеры, `local_llm.sh`.
- `data/test/` — синтетические тикеты; `data/research/` — скачанные открытые данные (не в git).

## LLM для описаний кластеров
Нужен только для описаний кластеров (ТЗ: «если очень хочется заморочиться»). Код ходит в любой OpenAI-совместимый
endpoint, пресеты в `configs/`: локальные **llama.cpp** (`tools/local_llm.sh`, Qwen3-4B на CPU) и **Ollama**, бесплатные
API-тарифы Gemini / Groq / Cerebras / OpenRouter (ключ в `LLM_API_KEY`), корпоративный vLLM. Реальные данные заказчика —
только в локальную модель или корпоративный endpoint. Запуск на своём ПК с GPU — `docs/run-on-your-pc.md`.
