# Review-lurker — контекст проекта

Исследовательский прототип по ТЗ v2 (`docs/TZ.md`): (1) кластеризация комментариев к тикетам и влияние очистки,
(2) поиск аномальных звонков внутри тем. Результат — ноутбуки `research/0*.ipynb` и выводы `research/REPORT.md`.
Первая версия ТЗ (сервис) — `docs/TZ-v1-service.md`; её интерфейс и сценарий «вопрос к данным» живут в отделяемых
`ui/` и `src/lurker/service/`. Раздел «Площади квартир» из документа заказчика — чужая задача, не наша.
Роли: Андрей — менеджер (ставит задачи, принимает решения), ты — разработчик.
Отвечать по-русски, на «ты», компактно. Перед крупным шагом — коротко объяснить план.
Авторство: коммиты — от `GasaiYuno1 <42517112+GasaiYuno1@users.noreply.github.com>`, без приписок `Co-Authored-By`,
`Claude-Session` и «Generated with …» (в PR тоже); в файлах проекта не писать, что работу или проверки делал ИИ
(ручные проверки — «разработчик»). Работу вливать в `main`.

## ТЗ v2: ключевое
- Готовый сервис не нужен; нужен прототип + вывод по гипотезе, отрицательный вывод тоже засчитывается.
- Задача 1: вопросы — похожие в одной группе? сколько нерелевантных в кластере? понятное описание кластера? зависимость
  от предобработки (сравнить эмбеддинги сырого и очищенного текста)? Данные — открытые (banki.ru с Kaggle, Яндекс-отзывы),
  «замусорить» ссылкой, именем, телефоном (9–11 цифр). Реальный объём — 40 тыс. тикетов. LLM — по желанию, для описаний
  кластеров по 10–30 текстам.
- Задача 2: кластеры по темам → внутри темы расстояние до центра → топ-N → precision@K → вывод, порог, модель.
  Данные: `katarinaaaaa/evaluation-of-customer-service` (HF). Испорченные (шум STT) — не аномалии; аномальные — чужая
  тема, грубость, бред оператора; списки id. ~3000 звонков в день.

## Принятые решения
- Окружение: интернет открыт (huggingface, Kaggle без логина, PyPI, pytorch CPU). Нейросетевые эмбеддинги — через
  sentence-transformers, при несовместимом конфиге модели — запасная загрузка через transformers с явным пулингом
  (`lurker.embed._HFEncoder`: USER-base mean, USER-bge-m3 и FRIDA CLS). Скорость на 4 ядрах: `research/results/embed_timings.csv`.
- Тяжёлые расчёты — скриптами с кэшем в `build/research/` (не в git); ноутбуки только читают итоги из `research/results/`
  и собираются `research/make_notebooks.py`; выводы — в `research/conclusions_*.md` (подставляются в ноутбуки) и `research/REPORT.md`.
  Страница-итоги для заказчика — `research/report.html`, она же артефакт https://claude.ai/artifact/FnFSZgwpNUDqWmLnz75svu
  (в артефакт публикуется без строк `<!doctype>`/`<html>`/`<head>`/`<body>`); менять обе вместе.
- Очистка (`lurker.clean.RuleCleaner`) проверена на открытых отзывах: разбор структуры SMM только при признаках переписки
  («Мы:/Оператор:/Переписка:», строки «— »), метка «ФИО:» снимает только само имя, голые 9–12-значные номера удаляются;
  сохранность слов отзыва после «замусорить → очистить» 97–98% при нуле остатков ссылок, телефонов, email.
- Python 3.11, pandas. Ядро работает и **без LLM и без сети**: модели из pip-пакетов (natasha: NER + navec), эмбеддинги TF-IDF+SVD и navec, кластеры UMAP→HDBSCAN→c-TF-IDF (схема BERTopic). LLM — подключаемый слой через HTTP OpenAI-совместимого `/chat/completions` (`src/lurker/llm.py`, без SDK), endpoint/модель в `config.yaml` (ТЗ п. 5), ключ из `LLM_API_KEY`; ответы кэшируются в `.cache/llm`. Промпты LLM-слоёв покрыты тестами с `FakeLLM`, на живой модели не проверены.
- Главный сценарий — «сначала вопрос» (ТЗ п. 2, 8): промпт → период + понятие (`concept.py`: предмет запроса, пресеты, прототип, расширение по navec) → отбор по смыслу с порогом (`preclass.threshold`=0.5) → динамика понятия → сравнение периодов → тематики внутри выборки → уточняющий вопрос (`query.py`, `assistant.py`). Обзорные кластеры всего файла — вспомогательная вкладка, не центр продукта.
- Роли LLM в сценарии (по замеру на Qwen3-4B): период — всегда правила, LLM только если правила не нашли; критерии LLM *дополняют* понятие из промпта словами и примерами (`concept.augment_concept`), а не заменяют его — у слабых моделей критерии сами по себе хуже промпта; названия тем и саммари — там, где LLM даёт видимый выигрыш. Ключевые слова матчатся по основе (≥ 6 букв): «оскорбление» ловит «оскорблял».
- Очистка перед эмбеддингом обязательна (ТЗ п. 4): регулярки + NER для формального мусора и ПД, LLM few-shot поверх (ПД не уходят в API). Предклассификация — по смыслу (эмбеддинги + критерии), не регулярки.
- Динамика считается pandas'ом без LLM; период из промпта — правилами (`period.py`), LLM-вариант даёт тот же JSON. Относительные периоды — от последней даты в данных.
- По умолчанию темы не укрупняются (`cluster.max_topics: null`): на тестовых данных слияние по центроидам роняет чистоту 0.84 → 0.63. Ручка есть в интерфейсе и конфиге.
- **Бюджет на LLM — 0.** Пресеты `configs/*.yaml`: бесплатные API-тарифы Gemini (endpoint `generativelanguage.googleapis.com` доступен из этого окружения, нужен только ключ в `LLM_API_KEY`), Groq, Cerebras, OpenRouter; локальные Ollama / llama.cpp (`tools/local_llm.sh`, нужен доступ к huggingface.co) / vLLM. `lurker --config configs/<x>.yaml llmcheck` прогоняет все LLM-шаги на тестовых комментариях. Клиент держит `rate_limit_rpm`, уважает `Retry-After`, при 400 на `response_format` сам отключает json_mode, считает токены. Реальные данные заказчика во внешние API не отправлять.
- Тесты — stdlib `unittest`, без сети. Внешние пакеты — только из `requirements.txt`; `natasha`/`pymorphy2` ставятся из `requirements-nodeps.txt` с `--no-deps` (старый sdist `docopt` не собирается), остальное их окружение — в `requirements.txt`; исследование — `requirements-research.txt`; ставить через `python3 -m pip`.

## Ключевые результаты (подробно — `research/REPORT.md`; после аудита)
- Задача 1 — подтверждается при очистке, настройке и модели по длине текста. geo (12 рубрик, длинные): USER-base
  NMI 0.82 ± 0.004 / B³-P 0.83 / 87% чистых; тикеты (короткие): USER-bge-m3 0.86 ± 0.005 / B³-P 0.84 (USER-base там
  0.80, как гибрид и e5-base). Мусор ТЗ сам по себе 0.82 → 0.66, с SMM-форматом 0.56; очистка → 0.82 (ARI 0.93).
  На незнакомом формате (CRM-выгрузка, `dirty` уровень alt) правила не помогают (0.67) — аудит исправил правила и добавил
  `remove_boilerplate` (шаблоны по частоте в корпусе); на тикетах оно поднимает USER-base 0.80 → 0.84.
  BERTopic по умолчанию плох на коротких тикетах (102 кластера на 25 тем), на длинных отзывах терпим.
- Описания LLM (Qwen3-4B, 20 текстов): 38/39 названий совпадают с меткой, 1 частично; банки — ручная проверка 180
  текстов: 80% подходят (85% взвешенно), названия 8/8/2; 4 кластера на 90–100% об одном банке.
  Самооценка LLM («однородность», «выбивающиеся») почти наугад — не использовать. «О чём группа» — переспрос.
- Задача 2 — частично: звонок целиком не работает (AP 0.07); новизна центрированных реплик оператора внутри темы
  (USER-base, `turns_c`): AP 0.37 [0.30–0.47], P@40 0.53; вне выборки (Russian MultiWOZ, 3000) AP 0.57, P@40 0.63;
  вслепую в топ-40: 21 аномалия + 8 реально спорных звонков датасета; связь с оценками качества датасета AUC 0.57–0.66.
  Порог z ≥ 2: 3.5% / 6.1% помечено — на реальном потоке задавать объёмом (топ-N).
- Динамика (вопрос ТЗ «стали ли чаще жаловаться на домофон»): по кластерам по неделям найдено 5 из 7 заложенных в
  синтетику событий (мартовский всплеск ×7, рост оплаты/угроз/расторжений, зима), ряды повторяют настоящие (r 0.92–1.0);
  маленькая новая тема (23 тикета) растворяется в соседнем кластере — её ловит `lurker.novelty.emerging_groups`
  (месяц против всего прошлого; группа новых текстов, у которой и центр далёк от прошлого): найдена в первый же месяц,
  ложных кандидатов 0 за 6 месяцев.
- Необычные комментарии (`lurker.novelty.unusual_scores`: kNN-новизна по всему корпусу, тексты от 4 слов): подмешанные
  50 необычных — AP 0.86 (контроль того же автора 0.08); на настоящих банках топ-20 — 4 необычных + 3 частично.
- Тональность (`lurker.sentiment`): на отзывах доля негатива по кластерам у blanchefort/rubert-base-cased-sentiment —
  r 0.99, ошибка 4 п.п.; на заметках операторов (тикеты) модели не работают (r ≤ 0.51) — тон кластера от LLM верен 13/13.
- Банки: удаление названий брендов (`configs/mask_banks.txt`) убирает «брендовые» кластеры (4 → 0), первая ступень
  делит на благодарности/жалобы; вторая ступень (`lurker.cluster.split_large`, кластеры > 15%) даёт карту тем.
- Звонки, подмена куска: новизна реплик клиента (AP подмены 0.19) и «чужая тема по репликам» (`offtopic_score`, 0.24)
  лучше основной (0.11), но в одной оценке с ней проигрывают — вести вторым списком.
- Воспроизводимость: в пустом контейнере по README данные, метрики и кластеры совпали до знака; все 89 тестов проходят
  в чистом venv.
- Готовые скрипты: `research/run_comments.py` (USER-bge-m3, удаление шаблонов, вторая ступень; `--sentiment`, `--mask`;
  выходы clusters/assignments/manual_check/dynamics/new_topics/unusual/boilerplate), `research/run_calls.py`.

## Структура
- `src/lurker/` (ядро): `clean` (`RuleCleaner`, `LLMCleaner`, `client_turns`, `remove_boilerplate`), `dirty` (`make_dirty`: уровни tz/smm/mix/alt),
  `novelty` (`unusual_scores`, `emerging_groups`), `sentiment` (`predict_sentiment`), `cluster.split_large`, `anomaly.offtopic_score`,
  `embed` (`get_embedder`, `ST_MODELS`, `SentenceTransformerEmbedder`, `_HFEncoder`), `cluster` (`cluster_embeddings`,
  `evaluate` — NMI/ARI/чистота/B³, `knn_overlap`, `knn_label_share`), `anomaly` (`assign_topics`, `robust_z`,
  `centroid/knn/lof/turn/coherence/relevance_score`, `turn_novelty`, `doc_max`, `split_dialog`, `ranking_metrics`),
  `clean` также `remove_boilerplate`, `dirty` уровень `alt` (незнакомые очистке форматы), `summarize` (`describe_cluster`, `name_topics`, …),
  `llm`, `config`, `text`, `data`, `dynamics`, `period`, `preclass`, `pipeline`, `cli`.
- `src/lurker/service/` (отделяемое): `concept`, `query`, `assistant` — сценарий «вопрос к данным» из ТЗ v1. `ui/` — Streamlit
  (`ui/app.py`, `ui/ui_smoke.py`, свои requirements).
- `research/`: `common.py` (пути, загрузка с кэшем); `comments/` — `prepare_data`, `embed`, `experiment`, `sensitivity`,
  `describe`, `eval_descriptions`, `dynamics_check`, `speed`, `scale_test`, `robustness` (аудит очистки и зёрен); `calls/` — `prepare_calls`,
  `run_anomaly`, `extra`, `bootstrap` (интервалы, оценки датасета), `multiwoz_check` (вне выборки);
  `run_comments.py`, `run_calls.py` — прогон на своих данных; `make_notebooks.py`; `results/` — итоговые таблицы;
  `01_comments_clustering.ipynb`, `02_anomalous_calls.ipynb`, `REPORT.md`, `report.html`, `conclusions_*.md`.
- `data/test/` — синтетические тикеты (seed 1, 3000 строк, скрытые темы); `data/research/` — открытые данные (не в git).
- `tests/` — `test_clean` (в т.ч. регрессии на открытых отзывах), `test_anomaly`, `test_analysis`, `test_core_standalone`,
  `test_service`, `test_gen`, `test_data`, `test_llm`, `test_period`, `test_run_scripts` (готовые скрипты на мини-данных).

## Команды
```
python3 -m pip install -r requirements.txt && python3 -m pip install --no-deps -r requirements-nodeps.txt && python3 -m pip install -e .
python3 -m unittest discover -s tests                      # тесты, ~1.5 мин
python3 research/comments/prepare_data.py                  # задача 1: данные (bank, geo, tickets × orig/dirty/clean)
python3 research/comments/embed.py [--only geo] [--models e5-small]   # эмбеддинги с кэшем (часы на CPU)
python3 research/comments/experiment.py                    # кластеры и метрики -> research/results/comments_metrics.csv
python3 research/comments/sensitivity.py                   # настройки UMAP/HDBSCAN -> sensitivity.csv
python3 research/comments/describe.py --dataset bank --variant clean --model user-base [--no-llm]   # описания кластеров
python3 research/calls/prepare_calls.py                    # задача 2: звонки + испорченные + аномальные, списки id
python3 research/calls/run_anomaly.py [--models ...] [--variants stt,raw]
python3 research/comments/eval_descriptions.py            # названия/«выбивающиеся» LLM против настоящих меток
python3 research/comments/dynamics_check.py               # динамика: ловят ли кластеры заложенные всплески и рост
python3 research/comments/novelty.py                       # новые темы по месяцам и необычные комментарии
python3 research/comments/sentiment.py                     # тональность: 4 модели против рейтинга/разметки
python3 research/comments/bank_brands.py                   # банки: без названий брендов + вторая ступень
python3 research/comments/scale_settings.py                # 40 тыс.: конвейер целиком и настройки под объём
python3 research/comments/robustness.py [prepare embed dirt seeds seeds_bp]   # аудит: мусор ТЗ отдельно, незнакомый мусор, зёрна
python3 research/calls/bootstrap.py                        # интервалы задачи 2 + связь с оценками качества датасета
python3 research/calls/multiwoz_check.py prepare <клон Russian-MultiWOZ> && ... run && ... eval   # вне выборки
python3 research/make_notebooks.py [--no-exec]             # ноутбуки из результатов
python3 research/run_comments.py <csv|xlsx> --text <колонка> [--date <колонка>] [--llm configs/llamacpp.yaml]   # свои данные
python3 research/run_calls.py <csv|xlsx> --text <колонка> [--id <колонка>] | --long --id --speaker --text
tools/local_llm.sh                                         # локальная Qwen3-4B (llama.cpp) для описаний кластеров
streamlit run ui/app.py                                    # интерфейс (отделяемый)
```
