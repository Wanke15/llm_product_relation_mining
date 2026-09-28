from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

from .batch import batch_embed
from .providers import post_json


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize(vec):
    n = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
    return [float(x) / n for x in vec]


def _featurize(text: str, dim: int) -> list[float]:
    """特征哈希：字符 1/2/3-gram，确定性（依赖 hashlib，非 Python 随机化 hash）。"""
    vec = [0.0] * dim
    tokens = [text[i:i + n] for n in (1, 2, 3) for i in range(len(text) - n + 1)]
    if not tokens:
        tokens = [text or '<empty>']
    for tok in tokens:
        h = hashlib.sha256(tok.encode('utf-8')).digest()
        idx = int.from_bytes(h[:4], 'big') % dim
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign
    return normalize(vec)


class HashEmbedding:
    """零下载、确定性的本地 embedding。仅用于无密钥时打通召回流程。"""

    def __init__(self, dim=256):
        self.dim = dim
        self.model = f'hash-ngram-{dim}'
        self.revision = ''

    def embed(self, texts):
        return [_featurize(t.lower(), self.dim) for t in texts]


class OpenAIEmbedding:
    """OpenAI-compatible embeddings 适配器。

    同步（POST {model, input}）用于小批量；文本量达到 `EMBEDDING_BATCH_MIN` 时改走 Batch 文件接口（百炼 50% 费用）。
    """

    def __init__(self, url=None, key=None, model=None, revision='',
                 batch_min=None, batch_base=None, completion_window=None, batch_size=None):
        self.model = model or os.getenv('EMBEDDING_MODEL', '')
        self.key = key or os.getenv('EMBEDDING_API_KEY', '')
        self.url = url or os.getenv('EMBEDDING_URL', '')
        self.revision = revision or os.getenv('EMBEDDING_REVISION', '')
        self.batch_size = batch_size if batch_size is not None else int(os.getenv('EMBEDDING_BATCH_SIZE', '16'))
        self.batch_min = batch_min if batch_min is not None else int(os.getenv('EMBEDDING_BATCH_MIN', '0'))
        self.batch_base = batch_base or os.getenv('EMBEDDING_BATCH_BASE_URL', '') or self._derive_base(self.url)
        self.completion_window = completion_window or os.getenv('EMBEDDING_BATCH_COMPLETION_WINDOW', '24h')
        if not (self.model and self.key and self.url):
            raise ValueError('Set EMBEDDING_URL, EMBEDDING_API_KEY and EMBEDDING_MODEL in .env')

    @staticmethod
    def _derive_base(url):
        base = url.rstrip('/')
        return base[:-len('/embeddings')] if base.endswith('/embeddings') else base

    def embed(self, texts):
        if self.batch_min and len(texts) >= self.batch_min:
            return batch_embed(self.batch_base, self.key, self.model, texts,
                               self.completion_window, sync_fn=self._embed_sync)
        return self._embed_sync(texts)

    def _embed_sync(self, texts):
        vectors = []
        total = len(texts)
        done = 0
        next_mark = 10
        for i in range(0, total, self.batch_size):
            chunk = texts[i:i + self.batch_size]
            raw = post_json(self.url, self.key, {'model': self.model, 'input': chunk})
            data = raw.get('data')
            if not isinstance(data, list) or len(data) != len(chunk):
                raise RuntimeError('Embedding response missing/incomplete data')
            for item in data:
                v = item.get('embedding')
                if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                    raise RuntimeError('Embedding response invalid vector')
                vectors.append([float(x) for x in v])
            done += len(chunk)
            if total >= 20:  # 大批量才打点，小批量不刷屏
                pct = done * 100 // total
                if pct >= next_mark:
                    print(f'  embedding {done}/{total} ({pct}%)', flush=True)
                    next_mark = (pct // 10 + 1) * 10
        return vectors


def create_embedding() -> EmbeddingProvider:
    if os.getenv('EMBEDDING_MODEL') and os.getenv('EMBEDDING_API_KEY') and os.getenv('EMBEDDING_URL'):
        return OpenAIEmbedding()
    return HashEmbedding(int(os.getenv('EMBEDDING_DIM', '256')))