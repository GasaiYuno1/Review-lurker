#!/usr/bin/env bash
# Локальная LLM с OpenAI-совместимым API через llama.cpp, без видеокарты (бюджет 0).
# Нужен доступ к huggingface.co для скачивания GGUF (один раз). На 4 ядрах 4B-модель даёт ~8–12 токенов/с:
# хватает проверить промпты на десятках комментариев; для полных прогонов — Ollama на машине с GPU или бесплатный API.
# Использование: tools/local_llm.sh [репозиторий] [файл.gguf] [порт]
set -euo pipefail
REPO="${1:-Qwen/Qwen3-4B-GGUF}"
FILE="${2:-Qwen3-4B-Q4_K_M.gguf}"
PORT="${3:-8080}"
DIR="${LLM_DIR:-build/models}"
mkdir -p "$DIR"
python3 -c "import llama_cpp" 2>/dev/null || pip install -q llama-cpp-python huggingface_hub   # сборка ~5–10 мин
if [ ! -f "$DIR/$FILE" ]; then
  python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('$REPO', '$FILE', local_dir='$DIR')"
fi
echo "endpoint: http://localhost:$PORT/v1 (конфиг: configs/llamacpp.yaml)"
# chat-шаблон берётся из метаданных GGUF; n_batch побольше — быстрее обработка длинных промптов на CPU
exec python3 -m llama_cpp.server --model "$DIR/$FILE" --port "$PORT" --n_ctx 8192 --n_batch 512 --n_threads "$(nproc)" --n_threads_batch "$(nproc)" --verbose false
