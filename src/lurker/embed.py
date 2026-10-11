"""Эмбеддинги текстов. Бэкенды (config `embedding.backend`):
  tfidf   — леммы + символьные n-граммы → TF-IDF → SVD (LSA); без сети и моделей, всегда доступен;
  navec   — среднее словарных векторов navec (внутри пакета natasha, 300d, 250k слов) с IDF-весами;
  hybrid  — конкатенация tfidf и navec (по умолчанию);
  st      — sentence-transformers (модель из `embedding.model`; нужен доступ к huggingface или локальная копия).
Все бэкенды: fit(texts) на корпусе, encode(texts) → L2-нормированная матрица (косинус = скалярное произведение).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence

import numpy as np

from . import text as T


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, dim: int = 200, char: bool = True, min_df: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.dim = dim
        self.word = TfidfVectorizer(analyzer=T.analyzer, min_df=min_df, max_df=0.7, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=max(min_df, 3), max_df=0.7,
                                    sublinear_tf=True, max_features=150_000, lowercase=True) if char else None
        self.svd = None

    def _raw(self, texts: Sequence[str], fit: bool):
        from scipy.sparse import hstack
        parts = [self.word.fit_transform(texts) if fit else self.word.transform(texts)]
        if self.char is not None:
            parts.append(self.char.fit_transform(texts) if fit else self.char.transform(texts))
        return hstack(parts).tocsr()

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        from sklearn.decomposition import TruncatedSVD
        X = self._raw(texts, fit=True)
        k = max(2, min(self.dim, X.shape[1] - 1, len(texts) - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.svd.fit(X)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("сначала fit()")
        return _norm(self.svd.transform(self._raw(texts, fit=False)))


class NavecEmbedder:
    name = "navec"

    def __init__(self):
        from natasha import NewsEmbedding
        self.navec = NewsEmbedding()  # объект Navec: `w in navec`, navec[w], navec.pq.dim
        self.idf: dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> "NavecEmbedder":
        df: Counter = Counter()
        for t in texts:
            df.update(set(self._words(t)))
        n = max(1, len(texts))
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        return self

    def _words(self, text: str) -> list[str]:
        out = []
        for tok in T.tokens(text):
            if tok in T.STOPWORDS or len(tok) < 2 or tok.isdigit():
                continue
            l = T.lemma(tok)
            out.append(l if l in self.navec else tok)
        return [w for w in out if w in self.navec]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        dim = self.navec.pq.dim
        M = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            acc, wsum = np.zeros(dim, dtype=np.float32), 0.0
            for w in self._words(t):
                k = self.idf.get(w, 1.0)
                acc += k * self.navec[w]
                wsum += k
            if wsum:
                M[i] = acc / wsum
        return _norm(M)


class HybridEmbedder:
    name = "hybrid"

    def __init__(self, dim: int = 200, navec_weight: float = 1.0):
        self.tfidf = TfidfEmbedder(dim=dim)
        self.navec = NavecEmbedder()
        self.w = navec_weight

    def fit(self, texts: Sequence[str]) -> "HybridEmbedder":
        self.tfidf.fit(texts)
        self.navec.fit(texts)
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return _norm(np.hstack([self.tfidf.encode(texts), self.w * self.navec.encode(texts)]))


# Нейросетевые модели для экспериментов (ТЗ v2): короткое имя → (id на huggingface, префикс, лимит токенов).
# Префиксы — по карточкам моделей: e5/USER-base — «query: » для симметричных задач; FRIDA — «categorize_topic: »
# для группировки по теме; rubert-tiny2 и USER-bge-m3 — без префикса.
ST_MODELS: dict[str, tuple[str, str, int]] = {
    "rubert-tiny2": ("cointegrated/rubert-tiny2", "", 512),
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", 512),
    "e5-base": ("intfloat/multilingual-e5-base", "query: ", 512),
    "user-base": ("deepvk/USER-base", "query: ", 512),
    "user-bge-m3": ("deepvk/USER-bge-m3", "", 512),
    "frida": ("ai-forever/FRIDA", "categorize_topic: ", 512),
}
# пулинг для запасной загрузки через transformers (если конфиг sentence-transformers модели не читается)
POOLING = {"deepvk/USER-base": "mean", "deepvk/USER-bge-m3": "cls", "ai-forever/FRIDA": "cls"}


class _HFEncoder:
    """Минимальная замена SentenceTransformer: AutoModel + пулинг (mean/cls) + L2. Для T5 (FRIDA) — только энкодер."""

    def __init__(self, hf_id: str, device: str = "cpu", pooling: str = "mean"):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        cfg = AutoConfig.from_pretrained(hf_id)
        if getattr(cfg, "model_type", "") == "t5":
            from transformers import T5EncoderModel
            self.model = T5EncoderModel.from_pretrained(hf_id)
        else:
            self.model = AutoModel.from_pretrained(hf_id)
        self.model.eval().to(device)
        self.device = device
        self.pooling = pooling
        self.max_seq_length = 512

    def encode(self, texts, batch_size: int = 32, **kw) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = self.tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                                 max_length=self.max_seq_length, return_tensors="pt").to(self.device)
                h = self.model(**batch).last_hidden_state
                if self.pooling == "cls":
                    v = h[:, 0]
                else:
                    m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                out.append(v.float().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)


class SentenceTransformerEmbedder:
    name = "st"

    def __init__(self, model: str, device: str = "cpu", batch_size: int = 32, prefix: Optional[str] = None,
                 max_seq_length: Optional[int] = None):
        from sentence_transformers import SentenceTransformer  # импорт здесь, чтобы не тянуть torch без надобности
        hf_id, default_prefix, default_len = ST_MODELS.get(model, (model, "query: " if "e5" in model.lower() else "", 512))
        self.name = model
        self.hf_id = hf_id
        try:
            if hf_id in POOLING:
                raise TypeError("явный пулинг")  # у этих моделей надёжнее собрать пулинг самим
            self.model = SentenceTransformer(hf_id, device=device)
        except (TypeError, ValueError, KeyError, OSError) as e:
            if isinstance(e, OSError) and "does not appear to have" not in str(e):
                raise
            self.model = _HFEncoder(hf_id, device=device, pooling=POOLING.get(hf_id, "mean"))
        self.model.max_seq_length = max_seq_length or default_len
        self.batch_size = batch_size
        self.prefix = default_prefix if prefix is None else prefix

    def fit(self, texts: Sequence[str]):
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # сортировка по длине — меньше паддинга, заметно быстрее на CPU
        order = np.argsort([len(t) for t in texts])
        inputs = [self.prefix + texts[i] for i in order]
        if isinstance(self.model, _HFEncoder):
            E = self.model.encode(inputs, batch_size=self.batch_size)
        else:
            E = self.model.encode(inputs, batch_size=self.batch_size, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)
        out = np.empty_like(E)
        out[order] = E
        return _norm(out)


def get_embedder(backend: str = "hybrid", model: Optional[str] = None, device: str = "cpu", dim: int = 200):
    if backend == "tfidf":
        return TfidfEmbedder(dim=dim)
    if backend == "navec":
        return NavecEmbedder()
    if backend == "hybrid":
        try:
            return HybridEmbedder(dim=dim)
        except Exception:
            return TfidfEmbedder(dim=dim)
    if backend == "st":
        return SentenceTransformerEmbedder(model or "e5-small", device=device)
    if backend in ST_MODELS:
        return SentenceTransformerEmbedder(backend, device=device)
    raise ValueError(f"неизвестный бэкенд эмбеддингов: {backend}")


def _norm(M) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n
