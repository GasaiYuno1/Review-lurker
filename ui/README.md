# Интерфейс (отделяемый)

Streamlit-приложение для ручных проверок и визуала поверх ядра `lurker`: загрузка CSV/XLSX, вопрос к данным
(сценарий из первой версии ТЗ, подпакет `lurker.service`), обзор тем и их динамики, данные с очищенным текстом.

```
pip install -r ui/requirements.txt
streamlit run ui/app.py
python3 ui/ui_smoke.py --chromium /opt/pw-browsers/chromium     # скриншоты в build/ui-*.png (нужен playwright)
```

Отделение: интерфейс и сценарий вопроса к данным не нужны для исследования по текущему ТЗ (`docs/TZ.md`).
Чтобы убрать их целиком, удалить `ui/`, `src/lurker/service/` и `tests/test_service.py`; ядро, скрипты `research/`
и остальные тесты от них не зависят (проверяется в `tests/test_core_standalone.py`).
