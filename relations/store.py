from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np

from .core import ROOT
from .document import embedding_text_hash, product_hash
from .gender import gender_of, load_gender_rules

SCHEMA_VERSION = 1

_PRODUCT_COLUMNS = ('spu_id', 'name', 'branch', 'category_group', 'category', 'subcategory',
                    'subcategory_id', 'features', 'style', 'theme', 'attribute_text', 'season',
                    'available_colors', 'price', 'image_url')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
  spu_id TEXT PRIMARY KEY,
  name TEXT, branch TEXT, category_group TEXT, category TEXT,
  subcategory TEXT, subcategory_id TEXT, features TEXT, style TEXT,
  theme TEXT, attribute_text TEXT, season TEXT,
  available_colors TEXT, price REAL, image_url TEXT,
  product_hash TEXT NOT NULL,
  embedding_text_hash TEXT,
  gender TEXT NOT NULL DEFAULT '中性',
  is_active INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS embeddings (
  cache_key TEXT PRIMARY KEY,
  kind TEXT NOT NULL,             -- 'product' | 'query'
  text_hash TEXT NOT NULL,
  model TEXT NOT NULL,
  revision TEXT NOT NULL DEFAULT '',
  dim INTEGER NOT NULL,
  vector BLOB NOT NULL,           -- float32 原始字节
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS index_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  model TEXT, revision TEXT, dim INTEGER, count INTEGER, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS candidates (
  pair_key TEXT NOT NULL,
  channel TEXT NOT NULL,
  anchor_spu_id TEXT NOT NULL,
  candidate_spu_id TEXT NOT NULL,
  rank INTEGER,
  retrieval_score REAL,
  retrieval_version TEXT,
  PRIMARY KEY (pair_key, channel)
);
CREATE INDEX IF NOT EXISTS idx_candidates_anchor ON candidates(anchor_spu_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_kind ON embeddings(kind, model, revision);
CREATE TABLE IF NOT EXISTS recall_report (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  json TEXT
);
CREATE TABLE IF NOT EXISTS pairs (
  pair_key TEXT PRIMARY KEY,
  a TEXT NOT NULL, b TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screen (
  pair_key TEXT PRIMARY KEY,
  a_hash TEXT NOT NULL, b_hash TEXT NOT NULL,
  status TEXT NOT NULL,             -- ok | insufficient_input | error
  probability_related REAL,
  model TEXT, prompt_version TEXT, rule_hash TEXT, config_hash TEXT,
  usage TEXT, latency_ms INTEGER, raw_response TEXT, error TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS details (
  pair_key TEXT PRIMARY KEY,
  a_hash TEXT NOT NULL, b_hash TEXT NOT NULL,
  status TEXT NOT NULL,             -- ok | error
  route TEXT,                       -- normal | audit | forced
  result TEXT, label TEXT,
  model TEXT, prompt_version TEXT, rubric_hash TEXT, config_hash TEXT,
  usage TEXT, latency_ms INTEGER, raw_response TEXT, error TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
  run_id TEXT PRIMARY KEY,
  created_at TEXT, config_hash TEXT, screen_rule_hash TEXT, rubric_hash TEXT,
  status TEXT, counts TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
  pair_key TEXT PRIMARY KEY,
  review TEXT, updated_at TEXT
);
"""


def db_path():
    return ROOT / os.getenv('DB_PATH', 'data/relations.db')


class Store:
    def __init__(self, path=None):
        self.path = path or db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.migrate()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.conn.close()

    def migrate(self):
        self.conn.executescript(_SCHEMA)
        self._ensure_column('products', 'gender', "TEXT NOT NULL DEFAULT '中性'")
        self.conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
        self.conn.commit()

    def _ensure_column(self, table, column, decl):
        cols = {r['name'] for r in self.conn.execute(f'PRAGMA table_info({table})')}
        if column not in cols:
            self.conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {decl}')

    # ---- products ----
    def upsert_products(self, products):
        now = datetime.now(timezone.utc).isoformat()
        rules, _ = load_gender_rules()
        rows = []
        for p in products:
            active = 1 if (p.get('branch') != '后勤' and p.get('name')) else 0
            gender = gender_of(p.get('branch'), p.get('category_group'), rules)
            rows.append((
                p['spu_id'], p.get('name'), p.get('branch'), p.get('category_group'), p.get('category'),
                p.get('subcategory'), p.get('subcategory_id'), p.get('features'), p.get('style'), p.get('theme'),
                p.get('attribute_text'), p.get('season'),
                json.dumps(p.get('available_colors') or [], ensure_ascii=False),
                p.get('price'), p.get('image_url'),
                product_hash(p), embedding_text_hash(p), gender, active, now,
            ))
        self.conn.executemany(
            f'INSERT OR REPLACE INTO products ({", ".join(_PRODUCT_COLUMNS)}, product_hash, embedding_text_hash, gender, is_active, updated_at) '
            f'VALUES ({", ".join("?" for _ in _PRODUCT_COLUMNS)}, ?, ?, ?, ?, ?)', rows)
        self.conn.commit()

    def row_to_product(self, row):
        p = {c: row[c] for c in _PRODUCT_COLUMNS}
        p['available_colors'] = json.loads(row['available_colors'] or '[]')
        p['price'] = row['price']
        p['gender'] = row['gender']
        return p

    def product(self, spu_id):
        row = self.conn.execute('SELECT * FROM products WHERE spu_id = ?', (spu_id,)).fetchone()
        return self.row_to_product(row) if row else None

    def active_products(self):
        rows = self.conn.execute('SELECT * FROM products WHERE is_active = 1').fetchall()
        return [self.row_to_product(r) for r in rows]

    def active_ids(self):
        return [r['spu_id'] for r in self.conn.execute('SELECT spu_id FROM products WHERE is_active = 1')]

    # ---- embeddings ----
    def embedding_hit(self, cache_key, text_hash):
        row = self.conn.execute('SELECT * FROM embeddings WHERE cache_key = ?', (cache_key,)).fetchone()
        return row is not None and row['text_hash'] == text_hash

    def get_vector(self, cache_key) -> np.ndarray | None:
        row = self.conn.execute('SELECT * FROM embeddings WHERE cache_key = ?', (cache_key,)).fetchone()
        return np.frombuffer(row['vector'], dtype=np.float32) if row else None

    def put_embedding(self, cache_key, kind, text_hash, model, revision, dim, vector, replace_kind=None):
        if replace_kind:
            self.conn.execute('DELETE FROM embeddings WHERE kind = ? AND cache_key = ?', (replace_kind, cache_key))
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        self.conn.execute(
            'INSERT OR REPLACE INTO embeddings (cache_key, kind, text_hash, model, revision, dim, vector, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (cache_key, kind, text_hash, model, revision, int(dim), blob,
             datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def embedding_meta(self):
        return [dict(r) for r in self.conn.execute('SELECT cache_key, kind, text_hash, model, revision, dim FROM embeddings')]

    # ---- index state ----
    def get_index_state(self):
        row = self.conn.execute('SELECT * FROM index_state WHERE id = 1').fetchone()
        return dict(row) if row else None

    def set_index_state(self, model, revision, dim, count):
        self.conn.execute(
            'INSERT OR REPLACE INTO index_state (id, model, revision, dim, count, updated_at) VALUES (1, ?, ?, ?, ?, ?)',
            (model, revision, int(dim), int(count), datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    # ---- candidates ----
    def replace_candidates(self, rows):
        self.conn.execute('DELETE FROM candidates')
        self.conn.executemany(
            'INSERT OR REPLACE INTO candidates (pair_key, channel, anchor_spu_id, candidate_spu_id, rank, retrieval_score, retrieval_version) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)', rows)
        self.conn.commit()

    def candidate_rows(self):
        return [dict(r) for r in self.conn.execute('SELECT * FROM candidates ORDER BY anchor_spu_id')]

    def candidate_pair_count(self):
        return self.conn.execute('SELECT COUNT(DISTINCT pair_key) FROM candidates').fetchone()[0]

    def set_recall_report(self, report):
        self.conn.execute('INSERT OR REPLACE INTO recall_report (id, json) VALUES (1, ?)',
                          (json.dumps(report, ensure_ascii=False),))
        self.conn.commit()

    def recall_report(self):
        row = self.conn.execute('SELECT json FROM recall_report WHERE id = 1').fetchone()
        return json.loads(row['json']) if row else None

    def stats(self):
        products_total = self.conn.execute('SELECT COUNT(*) FROM products').fetchone()[0]
        products_active = self.conn.execute('SELECT COUNT(*) FROM products WHERE is_active = 1').fetchone()[0]
        embedded = self.conn.execute("SELECT COUNT(*) FROM embeddings WHERE kind = 'product'").fetchone()[0]
        index_state = self.get_index_state()
        pairs = self.candidate_pair_count()
        edges = self.conn.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
        per_channel = {r['channel']: r['n'] for r in
                       self.conn.execute('SELECT channel, COUNT(DISTINCT pair_key) AS n FROM candidates GROUP BY channel')}
        return dict(products_total=products_total, products_active=products_active, embedded=embedded,
                    index_state=index_state, pairs=pairs, edges=edges,
                    per_channel=per_channel)

    # ---- 两阶段（screen / detail）----
    def refresh_pairs(self):
        self.conn.execute('DELETE FROM pairs')
        rows = self.conn.execute('SELECT pair_key, anchor_spu_id, candidate_spu_id FROM candidates').fetchall()
        gathered = {}
        for r in rows:
            gathered.setdefault(r['pair_key'], set()).update((r['anchor_spu_id'], r['candidate_spu_id']))
        self.conn.executemany('INSERT INTO pairs (pair_key, a, b) VALUES (?, ?, ?)',
                              [(k, sorted(v)[0], sorted(v)[-1]) for k, v in gathered.items()])
        self.conn.commit()

    def distinct_pairs(self, limit=None, offset=0):
        sql = 'SELECT pair_key, a, b FROM pairs ORDER BY pair_key'
        if limit:
            sql += f' LIMIT {int(limit)} OFFSET {int(offset)}'
        return [dict(r) for r in self.conn.execute(sql)]

    def pairs_count(self):
        return self.conn.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]

    def search_pairs(self, q, limit=50, offset=0):
        like = f'%{q}%'
        rows = self.conn.execute(
            'SELECT DISTINCT p.pair_key, p.a, p.b FROM pairs p '
            'JOIN products a ON a.spu_id = p.a JOIN products b ON b.spu_id = p.b '
            'WHERE a.name LIKE ? OR b.name LIKE ? OR p.a = ? OR p.b = ? '
            'ORDER BY p.pair_key LIMIT ? OFFSET ?',
            (like, like, q.strip(), q.strip(), limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def pairs_merged(self, pairs=None, limit=50, offset=0):
        pairs = pairs if pairs is not None else self.distinct_pairs(limit=limit, offset=offset)
        out = []
        for p in pairs:
            pa, pb = self.product(p['a']), self.product(p['b'])
            channels = [dict(r) for r in self.conn.execute(
                'SELECT channel, rank, retrieval_score, retrieval_version FROM candidates WHERE pair_key=? ORDER BY channel, rank',
                (p['pair_key'],))]
            sc, dt = self.get_screen(p['pair_key']), self.get_detail(p['pair_key'])
            if sc:
                sc = dict(sc); sc.pop('raw_response', None); sc.pop('usage', None)
            if dt:
                dt = dict(dt); dt.pop('raw_response', None); dt.pop('usage', None)
                if dt.get('result'):
                    dt['result'] = json.loads(dt['result'])
            out.append(dict(pair_key=p['pair_key'], a=p['a'], b=p['b'],
                            a_name=pa['name'] if pa else None, b_name=pb['name'] if pb else None,
                            a_branch=pa['branch'] if pa else None, b_branch=pb['branch'] if pb else None,
                            a_cat=pa['category_group'] if pa else None, b_cat=pb['category_group'] if pb else None,
                            channels=channels, screen=sc, detail=dt, review=self.review(p['pair_key'])))
        return out

    def all_hashes(self):
        return {r['spu_id']: r['product_hash'] for r in self.conn.execute('SELECT spu_id, product_hash FROM products')}

    def screen_hit(self, pair_key, a_hash, b_hash, rule_hash, config_hash):
        row = self.conn.execute('SELECT * FROM screen WHERE pair_key = ?', (pair_key,)).fetchone()
        return (row is not None and row['status'] != 'error'
                and row['a_hash'] == a_hash and row['b_hash'] == b_hash
                and row['rule_hash'] == rule_hash and row['config_hash'] == config_hash)

    def get_screen(self, pair_key):
        row = self.conn.execute('SELECT * FROM screen WHERE pair_key = ?', (pair_key,)).fetchone()
        return dict(row) if row else None

    def put_screen(self, record):
        self.conn.execute(
            'INSERT OR REPLACE INTO screen (pair_key, a_hash, b_hash, status, probability_related, model, prompt_version, rule_hash, config_hash, usage, latency_ms, raw_response, error, created_at) '
            'VALUES (:pair_key, :a_hash, :b_hash, :status, :probability_related, :model, :prompt_version, :rule_hash, :config_hash, :usage, :latency_ms, :raw_response, :error, :created_at)', record)
        self.conn.commit()

    def detail_hit(self, pair_key, a_hash, b_hash, rubric_hash, config_hash):
        row = self.conn.execute('SELECT * FROM details WHERE pair_key = ?', (pair_key,)).fetchone()
        return (row is not None and row['status'] == 'ok'
                and row['a_hash'] == a_hash and row['b_hash'] == b_hash
                and row['rubric_hash'] == rubric_hash and row['config_hash'] == config_hash)

    def get_detail(self, pair_key):
        row = self.conn.execute('SELECT * FROM details WHERE pair_key = ?', (pair_key,)).fetchone()
        return dict(row) if row else None

    def put_detail(self, record):
        self.conn.execute(
            'INSERT OR REPLACE INTO details (pair_key, a_hash, b_hash, status, route, result, label, model, prompt_version, rubric_hash, config_hash, usage, latency_ms, raw_response, error, created_at) '
            'VALUES (:pair_key, :a_hash, :b_hash, :status, :route, :result, :label, :model, :prompt_version, :rubric_hash, :config_hash, :usage, :latency_ms, :raw_response, :error, :created_at)', record)
        self.conn.commit()

    # ---- pipeline runs ----
    def create_run(self, run_id, config_hash, screen_rule_hash, rubric_hash):
        self.conn.execute('INSERT OR REPLACE INTO pipeline_runs (run_id, created_at, config_hash, screen_rule_hash, rubric_hash, status) VALUES (?, ?, ?, ?, ?, ?)',
                          (run_id, datetime.now(timezone.utc).isoformat(), config_hash, screen_rule_hash, rubric_hash, 'running'))
        self.conn.commit()

    def set_run_status(self, run_id, status, counts):
        self.conn.execute('UPDATE pipeline_runs SET status = ?, counts = ? WHERE run_id = ?',
                          (status, json.dumps(counts, ensure_ascii=False), run_id))
        self.conn.commit()

    def get_run(self, run_id):
        row = self.conn.execute('SELECT * FROM pipeline_runs WHERE run_id = ?', (run_id,)).fetchone()
        return dict(row) if row else None

    def latest_run(self):
        row = self.conn.execute('SELECT * FROM pipeline_runs ORDER BY created_at DESC LIMIT 1').fetchone()
        return dict(row) if row else None

    # ---- reviews（pipeline 人工标注，keyed by pair_key）----
    def review(self, pair_key):
        row = self.conn.execute('SELECT review FROM reviews WHERE pair_key = ?', (pair_key,)).fetchone()
        return json.loads(row['review']) if row else None

    def put_review(self, pair_key, review):
        self.conn.execute('INSERT OR REPLACE INTO reviews (pair_key, review, updated_at) VALUES (?, ?, ?)',
                          (pair_key, json.dumps(review, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def all_reviews(self):
        return {r['pair_key']: json.loads(r['review'])
                for r in self.conn.execute('SELECT pair_key, review FROM reviews')}

    # ---- pipeline 统计 ----
    def pipeline_stats(self):
        def cnt(table, where='1'):
            return self.conn.execute(f'SELECT COUNT(*) FROM {table} WHERE {where}').fetchone()[0]

        pairs_total = self.candidate_pair_count()
        screen = {r['status']: r['n'] for r in self.conn.execute('SELECT status, COUNT(*) n FROM screen GROUP BY status')}
        screen_total = sum(screen.values())
        routes = {r['route']: r['n'] for r in self.conn.execute('SELECT route, COUNT(*) n FROM details WHERE route IS NOT NULL GROUP BY route')}
        detail_ok = cnt('details', "status='ok'")
        detail_err = cnt('details', "status='error'")
        probs = [r['probability_related'] for r in self.conn.execute("SELECT probability_related FROM screen WHERE status='ok'")]
        hist = [0] * 10
        for p in probs:
            if p is not None:
                hist[min(9, int(p * 10))] += 1
        run = self.latest_run()
        return dict(
            pairs_total=pairs_total,
            screen=dict(total=screen_total, ok=screen.get('ok', 0),
                        insufficient_input=screen.get('insufficient_input', 0),
                        error=screen.get('error', 0), pending=max(0, pairs_total - screen_total)),
            screen_pass_rate=(screen.get('ok', 0) / screen_total) if screen_total else 0.0,
            prob_histogram=hist,
            details=dict(total=detail_ok + detail_err, ok=detail_ok, error=detail_err, routes=routes),
            tokens=self._tokens(),
            run_counts=(json.loads(run['counts']) if run and run.get('counts') else {}),
            run=(dict(run) if run else None),
        )

    def _tokens(self):
        keys = {'input_tokens': 0, 'output_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        for table in ('screen', 'details'):
            for r in self.conn.execute(f'SELECT usage FROM {table} WHERE usage IS NOT NULL'):
                try:
                    u = json.loads(r['usage'])
                    if isinstance(u, dict):
                        for k in keys:
                            if isinstance(u.get(k), (int, float)):
                                keys[k] += int(u[k])
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
        return keys