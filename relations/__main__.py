from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from . import document
from .core import ROOT, DIMENSIONS, candidates, digest, load_products, label
from .embedding import create_embedding
from .pipeline import Pipeline, preview_threshold
from .providers import create_judge
from .recall import CHANNELS, FUNCTIONAL, QUERY_TPL_VERSION, SIMILAR, STYLE, generate_candidates, load_roles
from .screen import create_screen
from .store import Store
from .vector_index import load_index, create_index


def load_env():
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, sep, value = line.partition('=')
            if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key.strip()):
                raise ValueError('Invalid .env assignment')
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def runs_dir():
    return ROOT / os.getenv('RUNS_DIR', 'runs')


def load_quotas(total_cap):
    """每通道配额；若总和与总配额不符则等比缩放，保证每商品候选总数有界。"""
    q = {SIMILAR: int(os.getenv('SIMILARITY_TOP_K', '10')),
         FUNCTIONAL: int(os.getenv('FUNCTIONAL_TOP_K', '10')),
         STYLE: int(os.getenv('STYLE_TOP_K', '10'))}
    if sum(q.values()) <= 0:
        q = {c: 10 for c in CHANNELS}
    if sum(q.values()) != total_cap:
        out, alloc, s = {}, 0, sum(q.values())
        for j, c in enumerate(CHANNELS):
            out[c] = round(q[c] * total_cap / s) if j < len(CHANNELS) - 1 else total_cap - alloc
            alloc += out[c]
        return out
    return q


def make_query_embedder(store, embedding, tpl_version):
    """带 DB 缓存的查询 embedding：描述哈希 + 模板版本 + 模型/版本决定缓存键。"""
    model = embedding.model
    revision = getattr(embedding, 'revision', '')

    def embed(texts):
        out, todo = [None] * len(texts), []
        for i, t in enumerate(texts):
            th = digest({'tpl': tpl_version, 'text': t})
            key = f"query:{th}:{model}:{revision}"
            v = store.get_vector(key)
            if v is not None and store.embedding_hit(key, th):
                out[i] = v
            else:
                todo.append((i, t, key, th))
        if todo:
            for (i, _t, key, th), vec in zip(todo, embedding.embed([t[1] for t in todo])):
                arr = np.asarray(vec, dtype=np.float32)
                store.put_embedding(key, 'query', th, model, revision, len(arr), arr)
                out[i] = arr
        return out
    return embed


def index(args):
    """同步商品、生成缺失向量、重建索引（embedding 走缓存，不重复调用）。"""
    products = load_products(ROOT / os.getenv('DATA_PATH', 'data/prod_df.csv'))
    embedding = create_embedding()
    model, revision = embedding.model, getattr(embedding, 'revision', '')
    with Store() as store:
        store.upsert_products(list(products.values()))
        active = store.active_products()
        vector_by_pid, todo, cached = {}, [], 0
        for p in active:
            pid, th = p['spu_id'], document.embedding_text_hash(p)
            key = f"product:{pid}:{th}:{model}:{revision}"
            v = store.get_vector(key)
            if v is not None and store.embedding_hit(key, th):
                vector_by_pid[pid], cached = v, cached + 1
            else:
                todo.append((pid, th, document.embedding_text(p), key))
        if todo:
            for (pid, th, _text, key), vec in zip(todo, embedding.embed([t[2] for t in todo])):
                arr = np.asarray(vec, dtype=np.float32)
                store.put_embedding(key, 'product', th, model, revision, len(arr), arr)
                vector_by_pid[pid] = arr
        pids = sorted(vector_by_pid)
        dim = len(vector_by_pid[pids[0]]) if pids else 0
        if dim:
            idx = create_index(dim)
            idx.add(pids, [vector_by_pid[pid] for pid in pids])
            idx.save(ROOT / os.getenv('INDEX_DIR', 'data/index'))
        store.set_index_state(model, revision, dim, len(pids))
    print(f"indexed {len(pids)} active products | cache hit {cached}, embedded {len(todo)} | dim={dim} model={model}", flush=True)


def retrieve(args):
    """生成候选（不调用 Jev/LLM）；--limit 限本次候选商品对数量。"""
    embedding = create_embedding()
    model, revision = embedding.model, getattr(embedding, 'revision', '')
    roles, roles_ver = load_roles()
    total_cap = args.top_k if args.top_k is not None else int(os.getenv('CANDIDATES_PER_PRODUCT', '30'))
    quotas = load_quotas(total_cap)
    index_dir = ROOT / os.getenv('INDEX_DIR', 'data/index')
    if not (index_dir / 'meta.json').exists():
        raise FileNotFoundError('No index found; run: uv run python -m relations index')
    idx = load_index(index_dir)
    retrieval_version = f"model:{model}:{revision}:tpl:{QUERY_TPL_VERSION}:roles:{roles_ver}"
    with Store() as store:
        state = store.get_index_state()
        if state and (state.get('model') != model or state.get('revision') != revision or state.get('dim') != idx.dim):
            raise ValueError('Index built with different embedding config; re-run: uv run python -m relations index')
        by_id = {p['spu_id']: p for p in store.active_products()}
        if any(pid not in by_id for pid in idx.ids):
            raise ValueError('Index references inactive products; re-run: uv run python -m relations index')
        anchors = sorted(idx.ids)
        if args.limit:
            anchors = anchors[:max(1, math.ceil(args.limit / total_cap))]
        embed = make_query_embedder(store, embedding, QUERY_TPL_VERSION)
        rows, report = generate_candidates(by_id, idx, embed, roles, quotas, retrieval_version,
                                           fetch_k=total_cap, anchors=anchors)
        if args.limit:
            keep, limited = set(), []
            for r in rows:
                if r['pair_key'] in keep or len(keep) < args.limit:
                    keep.add(r['pair_key']); limited.append(r)
            rows = limited
        store.replace_candidates([(r['pair_key'], r['channel'], r['anchor_spu_id'], r['candidate_spu_id'],
                                   r['rank'], r['retrieval_score'], r['retrieval_version']) for r in rows])
        store.set_recall_report(report)
        store.refresh_pairs()
    cov = report['coverage']
    print(f"candidates: {len(rows)} channel-edges / {report['pairs']} unique pairs from {report['anchors']} anchors", flush=True)
    print(f"coverage: similar={cov[SIMILAR]} functional={cov[FUNCTIONAL]} style={cov[STYLE]} (of {report['anchors']} anchors)", flush=True)


def _detail_provider():
    provider = os.getenv('DETAIL_PROVIDER', 'llm')
    if provider == 'llm' and not (os.getenv('LLM_API_KEY') and os.getenv('LLM_MODEL')):
        return 'demo'
    return provider


def pipeline(args):
    """两阶段流水线：Jev Noul 初筛 → 路由 → LLM 精判。--dry-run 只预览规模不调用。"""
    rubric = read(ROOT / 'prompts/rubric.json')
    screen = create_screen()
    detail = create_judge(_detail_provider(), rubric)
    with Store() as store:
        pipe = Pipeline(store, screen, detail, rubric)
        if args.dry_run:
            ps = store.pipeline_stats()
            route = preview_threshold(store, pipe.threshold, pipe.audit_rate, pipe.seed)
            print('pipeline dry-run (no model calls)')
            print(f"  candidates: {ps['pairs_total']} unique pairs | screen done {ps['screen']['total']} | detail done {ps['details']['total']}")
            print(f"  screen pending: {ps['screen']['pending']} | detail ok: {ps['details']['ok']}")
            print(f"  screen={getattr(screen, 'model', '?')} detail={detail.model} threshold={pipe.threshold} audit_rate={pipe.audit_rate}")
            print(f"  route preview @threshold: {route}")
            return
        counts, run_id = pipe.run(limit=args.limit, llm_cap=args.llm_cap, resume=args.resume)
        print(f"pipeline run {run_id}")
        print(f"  screen: called {counts['screen_called']} reused {counts['screen_reused']} "
              f"error {counts['screen_error']} insufficient {counts['screen_insufficient']}")
        print(f"  route: normal {counts['normal']} audit {counts['audit']} screened_out {counts['screened_out']}")
        print(f"  detail: called {counts['detail_called']} reused {counts['detail_reused']} "
              f"ok {counts['detail_ok']} error {counts['detail_error']}")
        print('Review: uv run python -m relations serve', flush=True)


def run(args):
    base = runs_dir()
    if args.resume:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.resume):
            raise ValueError('Invalid run ID')
        directory = base / args.resume
        manifest = read(directory / 'manifest.json')
        products = read(directory / 'products.json')
        pairs = read(directory / 'pairs.json')
        rubric = read(directory / 'rubric.json')
        if args.provider and args.provider != manifest['provider']:
            raise ValueError('Cannot change provider on resume; start a new run')
    else:
        products = load_products(ROOT / os.getenv('DATA_PATH', 'data/prod_df.csv'))
        limit = args.limit if args.limit is not None else int(os.getenv('PAIR_LIMIT', '200'))
        if limit < 1:
            raise ValueError('Limit must be positive')
        if args.pairs:
            with open(args.pairs, encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            pairs, seen = [], set()
            for row in rows:
                a, b = sorted((row['a'].strip(), row['b'].strip()))
                if a == b or a not in products or b not in products:
                    raise ValueError('Invalid pair: ' + a + '/' + b)
                if (a, b) not in seen:
                    pairs.append(dict(pair_id=digest([a, b])[:20], a=a, b=b, source='custom'))
                    seen.add((a, b))
            pairs = pairs[:limit]
        else:
            pairs = candidates(products, limit, int(os.getenv('RANDOM_SEED', '42')))
        rubric = read(ROOT / 'prompts/rubric.json')
        if not pairs:
            raise ValueError('No candidate pairs')
        provider = args.provider or os.getenv('RELATION_PROVIDER', 'demo')
        judge = create_judge(provider, rubric)  # Fail before writing a run if config is invalid.
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
        directory = base / run_id
        manifest = dict(run_id=run_id, created_at=datetime.now(timezone.utc).isoformat(), provider=provider,
                        model=judge.model, rubric_hash=digest(rubric), data_hash=digest(products),
                        prompt_version=rubric['version'], image_input=False,
                        requested_pairs=limit, pair_count=len(pairs), population=len(products),
                        seed=int(os.getenv('RANDOM_SEED', '42')))
        for name, value in [('manifest', manifest), ('products', products), ('pairs', pairs), ('rubric', rubric)]:
            save(directory / (name + '.json'), value)
    judge = create_judge(manifest['provider'], rubric)
    if judge.model != manifest['model']:
        raise ValueError('Configured model differs from original run; start a new run')
    endpoint = os.getenv('LLM_URL', 'https://api.openai.com/v1/chat/completions') if manifest['provider'] == 'llm' else os.getenv('JEV_URL', 'https://api.typesafe.ai/v1/systemone')
    # Hash config to protect resume semantics without recording credentials or private URLs.
    config_hash = digest([endpoint, os.getenv('LLM_RESPONSE_FORMAT', 'json_object')])
    if manifest.get('config_hash', config_hash) != config_hash:
        raise ValueError('Endpoint/format changed; start a new run')
    manifest['config_hash'] = config_hash
    manifest['status'] = 'running'
    save(directory / 'manifest.json', manifest)
    results_dir = directory / 'results'
    results_dir.mkdir(exist_ok=True)
    print(f"Run: {manifest['run_id']} | {manifest['provider']} | {len(pairs)} pairs", flush=True)
    for i, pair in enumerate(pairs):
        out = results_dir / (pair['pair_id'] + '.json')
        if out.exists() and read(out)['status'] == 'ok':
            continue
        start = time.monotonic()
        record = dict(pair, run_id=manifest['run_id'])
        try:
            result, raw = judge.judge(products[pair['a']], products[pair['b']])
            record.update(status='ok', result=result, raw=raw, label=label(result))
        except Exception as exc:
            # Whitelist operational error text; never persist arbitrary provider payloads.
            error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__ + ': invalid response or input'
            record.update(status='error', error=error)
        record['latency_ms'] = round((time.monotonic() - start) * 1000)
        save(out, record)
        print(f"[{i+1}/{len(pairs)}] {pair['a']} / {pair['b']}: {record['status']}", flush=True)
    records = [read(p) for p in results_dir.glob('*.json')]
    manifest.update(status='complete' if all(r['status'] == 'ok' for r in records) else 'completed_with_errors',
                    ok=sum(r['status'] == 'ok' for r in records), errors=sum(r['status'] != 'ok' for r in records))
    save(directory / 'manifest.json', manifest)
    print('Review: uv run python -m relations serve', flush=True)
    if manifest['errors']:
        raise SystemExit(1)


def serve(args):
    base = runs_dir().resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, data, status=200, content_type='application/json; charset=utf-8'):
            body = data.encode() if isinstance(data, str) else json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def directory(self, run_id):
            if not re.fullmatch(r'[a-zA-Z0-9_-]+', run_id):
                raise ValueError('Invalid run ID')
            return base / run_id

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == '/':
                    return self.send((ROOT / 'web/index.html').read_text(encoding='utf-8'), content_type='text/html; charset=utf-8')
                if path == '/api/runs':
                    return self.send([read(p) for p in sorted(base.glob('*/manifest.json'), reverse=True)])
                if path.startswith('/api/run/'):
                    directory = self.directory(path.removeprefix('/api/run/'))
                    records = [read(p) for p in sorted((directory / 'results').glob('*.json'))]
                    pairs = read(directory / 'pairs.json')
                    products = read(directory / 'products.json')
                    ids = {p[k] for p in pairs for k in ('a', 'b')}
                    review_file = directory / 'reviews.json'
                    return self.send(dict(manifest=read(directory / 'manifest.json'), products={i: products[i] for i in ids},
                                          records=records, reviews=read(review_file) if review_file.exists() else {}))
                if path == '/api/stats':
                    with Store() as store:
                        return self.send(dict(store.stats(), recall=store.recall_report()))
                if path == '/api/pipeline':
                    with Store() as store:
                        thr = float(os.getenv('SCREEN_THRESHOLD', '0.5'))
                        audit = float(os.getenv('SCREEN_AUDIT_RATE', '0.0'))
                        seed = int(os.getenv('RANDOM_SEED', '42'))
                        return self.send(dict(stats=store.pipeline_stats(),
                                              config=dict(threshold=thr, audit_rate=audit),
                                              route_preview=preview_threshold(store, thr, audit, seed)))
                if path == '/api/preview':
                    qs = parse_qs(urlparse(self.path).query)
                    thr = float(qs.get('threshold', [os.getenv('SCREEN_THRESHOLD', '0.5')])[0])
                    with Store() as store:
                        return self.send(dict(threshold=thr,
                                              route=preview_threshold(store, thr, float(os.getenv('SCREEN_AUDIT_RATE', '0.0')),
                                                                      int(os.getenv('RANDOM_SEED', '42')))))
                if path == '/api/pairs':
                    qs = parse_qs(urlparse(self.path).query)
                    limit = min(200, int(qs.get('limit', ['50'])[0]))
                    offset = max(0, int(qs.get('offset', ['0'])[0]))
                    q = qs.get('q', [''])[0].strip()
                    with Store() as store:
                        if q:
                            pairs = store.search_pairs(q, limit, offset)
                            rows = store.pairs_merged(pairs=pairs)
                            total = len(pairs)
                        else:
                            rows = store.pairs_merged(limit=limit, offset=offset)
                            total = store.pairs_count()
                        return self.send(dict(total=total, rows=rows))
                self.send({'error': 'Not found'}, 404)
            except (ValueError, FileNotFoundError):
                self.send({'error': 'Run not found or invalid'}, 404)

        def do_POST(self):
            # Local-only UI; reject cross-origin browser writes.
            if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{args.port}', f'http://localhost:{args.port}'):
                return self.send({'error': 'Forbidden origin'}, 403)
            try:
                if self.path not in ('/api/review', '/api/pipeline/review') or \
                        not self.headers.get('Content-Type', '').startswith('application/json'):
                    return self.send({'error': 'Invalid request'}, 400)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 100000:
                    raise ValueError('Invalid size')
                data = json.loads(self.rfile.read(size))
                review = data['review']
                if review['status'] not in ('agree', 'corrected', 'uncertain') or not isinstance(review.get('note', ''), str):
                    raise ValueError('Invalid review')
                if set(review.get('scores', {})) != set(DIMENSIONS):
                    raise ValueError('Missing review scores')
                if any(v is not None and (type(v) is not int or v not in range(4)) for v in review['scores'].values()):
                    raise ValueError('Invalid score')
                if self.path == '/api/review':
                    directory = self.directory(data['run_id'])
                    pair_id = data['pair_id']
                    if pair_id not in {p['pair_id'] for p in read(directory / 'pairs.json')}:
                        raise ValueError('Unknown pair')
                    path = directory / 'reviews.json'
                    reviews = read(path) if path.exists() else {}
                    reviews[pair_id] = dict(review, updated_at=datetime.now(timezone.utc).isoformat())
                    save(path, reviews)
                else:
                    with Store() as store:
                        store.put_review(data['pair_key'], review)
                self.send({'ok': True})
            except (ValueError, KeyError, FileNotFoundError, TypeError):
                self.send({'error': 'Invalid review'}, 400)

    print(f'http://127.0.0.1:{args.port} (Ctrl+C to stop)', flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


def main():
    load_env()
    parser = argparse.ArgumentParser(description='SPU relation mining and review')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--provider', choices=['demo', 'llm', 'jev'])
    p.add_argument('--limit', type=int)
    p.add_argument('--pairs', help='CSV with a,b SPU IDs; same file enables provider comparisons')
    p.add_argument('--resume', help='Run ID; skip successful pairs, retry failures')
    p = sub.add_parser('index')
    p = sub.add_parser('retrieve')
    p.add_argument('--top-k', type=int, help='每商品候选上限（默认 CANDIDATES_PER_PRODUCT）')
    p.add_argument('--limit', type=int, help='本次最多输出的候选商品对数量')
    p = sub.add_parser('pipeline')
    p.add_argument('--dry-run', action='store_true', help='仅预览候选规模与缓存命中')
    p.add_argument('--limit', type=int, help='本次处理的候选商品对数量上限')
    p.add_argument('--llm-cap', type=int, help='本轮 LLM 精判调用上限（防初筛通过率过高费用失控）')
    p.add_argument('--resume', help='恢复指定 run 未完成任务（已完成阶段跳过）')
    p = sub.add_parser('serve')
    p.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    commands = {'run': run, 'index': index, 'retrieve': retrieve, 'pipeline': pipeline, 'serve': serve}
    try:
        commands[args.command](args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
