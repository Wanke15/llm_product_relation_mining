from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    dim: int

    def add(self, ids, vectors): ...

    def vectors_of(self, ids) -> np.ndarray: ...

    def search_batch(self, vectors, k, exclude_per_row=None) -> list[list[tuple[str, float]]]: ...

    def save(self, directory): ...

    def load(self, directory): ...

    def __len__(self): ...


def normalize(vec):
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class NumpyIndex:
    """暴力余弦检索，向量按 L2 归一后存储，cosine = 内积。12k 规模毫秒级。

    作为无 hnswlib Python3.11 Windows wheel 的兜底；协议不变，以后可插回 hnswlib。
    """

    def __init__(self, dim):
        self.dim = int(dim)
        self.ids: list[str] = []
        self.matrix = np.empty((0, self.dim), dtype=np.float32)
        self._pos: dict[str, int] = {}

    def __len__(self):
        return len(self.ids)

    def add(self, ids, vectors):
        new = np.asarray(vectors, dtype=np.float32)
        if new.ndim == 1:
            new = new[None, :]
        if new.shape[1] != self.dim:
            raise ValueError(f'dimension mismatch: {new.shape[1]} != {self.dim}')
        norms = np.linalg.norm(new, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new = new / norms
        self.matrix = np.vstack([self.matrix, new]) if len(self.matrix) else new
        for i in ids:
            self._pos[i] = len(self.ids)
            self.ids.append(i)

    def vectors_of(self, ids):
        return np.stack([self.matrix[self._pos[i]] for i in ids]) if ids else \
            np.empty((0, self.dim), dtype=np.float32)

    def _best(self, scores, k, exclude):
        n = scores.shape[0]
        if n == 0:
            return []
        need = min(k + (len(exclude) if exclude else 0), n)
        idx = np.argpartition(-scores, need - 1)[:need]
        idx = idx[np.argsort(-scores[idx])]
        out = []
        for i in idx:
            if self.ids[i] in exclude:
                continue
            out.append((self.ids[i], float(scores[i])))
            if len(out) >= k:
                break
        return out

    def search_batch(self, vectors, k, exclude_per_row=None):
        q = np.asarray(vectors, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        out = []
        for start in range(0, len(q), 256):
            chunk = q[start:start + 256]
            scores = chunk @ self.matrix.T
            for j, row in enumerate(scores):
                ex = exclude_per_row[start + j] if exclude_per_row else None
                out.append(self._best(row, k, ex))
        return out

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / 'matrix.npy', self.matrix)
        (directory / 'ids.json').write_text(json.dumps(self.ids, ensure_ascii=False), encoding='utf-8')
        (directory / 'meta.json').write_text(json.dumps({'dim': self.dim, 'count': len(self.ids)}), encoding='utf-8')

    def load(self, directory):
        directory = Path(directory)
        self.matrix = np.load(directory / 'matrix.npy').astype(np.float32)
        self.ids = json.loads((directory / 'ids.json').read_text(encoding='utf-8'))
        self.dim = int(self.matrix.shape[1])
        self._pos = {i: j for j, i in enumerate(self.ids)}
        return self


def create_index(dim, backend=None):
    return NumpyIndex(dim)


def load_index(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
    return NumpyIndex(int(meta['dim'])).load(directory)