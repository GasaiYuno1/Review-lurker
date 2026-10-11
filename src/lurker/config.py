"""Конфиг: config.yaml -> dataclass'ы. Endpoint и модель LLM вынесены сюда по требованию ТЗ (п. 5).

Порядок поиска файла: аргумент -> переменная окружения LURKER_CONFIG -> ./config.yaml -> дефолты.
Ключ API в файл не пишется: берётся из переменной окружения, имя которой задано в llm.api_key_env.
Переменные LURKER_LLM_BASE_URL и LURKER_LLM_MODEL перекрывают значения из файла.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.0
    timeout_s: float = 60.0
    max_retries: int = 3
    batch_size: int = 20
    rate_limit_rpm: int = 0          # >0 — не чаще N запросов в минуту (бесплатные тарифы)
    json_mode: bool = True           # response_format=json_object; при 400 от endpoint отключается само
    max_tokens: int = 2048           # лимит ответа по умолчанию (локальным моделям нужен явно)
    extra_body: dict = field(default_factory=dict)   # добавки в тело запроса: {"chat_template_kwargs": {"enable_thinking": false}}
    strip_thinking: bool = True      # вырезать <think>…</think> из ответа (Qwen3, DeepSeek-R1 и т.п.)
    message_suffix: str = ""         # дописывается к последнему сообщению пользователя, например " /no_think" для Qwen3

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env) or None


@dataclass
class EmbeddingConfig:
    backend: str = "hybrid"          # tfidf | navec | hybrid | st (sentence-transformers, нужна модель)
    model: str = "intfloat/multilingual-e5-small"
    device: str = "cpu"
    batch_size: int = 64


@dataclass
class ClusterConfig:
    method: str = "umap_hdbscan"     # umap_hdbscan | bertopic | agglomerative
    min_cluster_size: Optional[int] = None   # None = max(8, sqrt(n)/2)
    max_topics: Optional[int] = None
    merge_threshold: Optional[float] = 0.9   # сходство центроидов, при котором кластеры сливаются
    outlier_threshold: float = 0.35


@dataclass
class PreclassConfig:
    threshold: float = 0.5


@dataclass
class CacheConfig:
    dir: str = ".cache"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    preclass: PreclassConfig = field(default_factory=PreclassConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


def _build(cls: type, raw: Any, section: str):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config: секция '{section}' должна быть словарём")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"config: неизвестные ключи в '{section}': {', '.join(unknown)}")
    return cls(**raw)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Читает YAML (если файл есть) и применяет переопределения из окружения."""
    candidates = [path, os.environ.get("LURKER_CONFIG"), "config.yaml"]
    raw: dict = {}
    for c in candidates:
        if c and Path(c).is_file():
            with open(c, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            break
        if c is path and path is not None:
            raise FileNotFoundError(f"config: файл не найден: {path}")
    cfg = Config(
        llm=_build(LLMConfig, raw.get("llm"), "llm"),
        embedding=_build(EmbeddingConfig, raw.get("embedding"), "embedding"),
        cluster=_build(ClusterConfig, raw.get("cluster"), "cluster"),
        preclass=_build(PreclassConfig, raw.get("preclass"), "preclass"),
        cache=_build(CacheConfig, raw.get("cache"), "cache"),
    )
    if os.environ.get("LURKER_LLM_BASE_URL"):
        cfg.llm.base_url = os.environ["LURKER_LLM_BASE_URL"]
    if os.environ.get("LURKER_LLM_MODEL"):
        cfg.llm.model = os.environ["LURKER_LLM_MODEL"]
    return cfg
