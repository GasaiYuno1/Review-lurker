# Запуск на своём компьютере (полные прогоны с локальной LLM)

Полные прогоны (эмбеддинги всех моделей, описания кластеров LLM на тысячах текстов) быстрее идут на машине с GPU.
Ниже — установка, проверка локальной LLM и команды.

## Один раз
```
git clone https://github.com/GasaiYuno1/Review-lurker && cd Review-lurker
python3 -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt && python3 -m pip install --no-deps -r requirements-nodeps.txt && python3 -m pip install -e .
python3 -m unittest discover -s tests                # ~1 мин, всё должно быть зелёным
```
`natasha` и `pymorphy2` ставятся из `requirements-nodeps.txt` без зависимостей: pymorphy2 тянет старый `docopt`, который
не собирается на Python 3.11+ (замена `docopt-ng` уже в `requirements.txt`).

LLM локально — Ollama (ollama.com, одна команда установки), затем:
```
ollama pull qwen3:8b          # 5 ГБ; на GPU 8–12 ГБ — 30–60 токенов/с. Варианты: gemma3:12b, qwen3:14b, t-lite
lurker --config configs/ollama.yaml llmcheck          # пинг + все LLM-шаги на 20 тестовых комментариях, должно быть 6/6
```
Модель в `configs/ollama.yaml` (`model: qwen3:8b`) поменять на ту, что скачал.

## Что гонять «много»
Полное исследование (эмбеддинги 8 моделей на трёх наборах — часы на CPU, минуты на GPU):
```
python3 -m pip install torch && python3 -m pip install -r requirements-research.txt   # torch с CUDA — по инструкции pytorch.org
python3 research/comments/prepare_data.py && python3 research/comments/embed.py && python3 research/comments/experiment.py
python3 research/calls/prepare_calls.py && python3 research/calls/run_anomaly.py
python3 research/comments/describe.py --dataset bank --variant clean --model user-base --mcs 20 --config configs/ollama.yaml
python3 research/make_notebooks.py
```

## Свои данные (мок или реальные тикеты и звонки)
Положить файл в `data/mock/` (папка в `.gitignore`: реальные ПД в git не попадают), затем:
```
python3 research/run_comments.py data/mock/tickets.xlsx --text "Комментарий" --date "Дата" --llm configs/ollama.yaml
python3 research/run_calls.py data/mock/calls.xlsx --text "Расшифровка" --id "ID"
```
Результаты — `build/run_comments/` и `build/run_calls/`. С реальными данными LLM — только локальная (Ollama/llama.cpp).
Интерфейс для визуальной проверки: `streamlit run ui/app.py`.
