from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .core import digest, label
from .screen import create_screen, screen_config_hash, screen_rule_hash


def detail_rubric_hash(rubric):
    return digest(rubric)


def detail_config_hash(detail):
    """后端配置版本（模型/端点/格式 + 实际 provider 模型），不含密钥。"""
    return digest([os.getenv('LLM_URL', 'https://api.openai.com/v1/chat/completions'),
                   os.getenv('LLM_MODEL', ''), os.getenv('LLM_RESPONSE_FORMAT', 'json_object'),
                   getattr(detail, 'model', '')])


def _sufficient(p):
    return bool(p.get('name') and (p.get('category') or p.get('category_group')))


def preview_threshold(store, threshold, audit_rate, seed):
    """仅用已存初筛概率计算路由分布，不发模型请求；调整阈值/抽查率时复用已存概率。"""
    out = {'normal': 0, 'audit': 0, 'screened_out': 0}
    for r in store.conn.execute("SELECT pair_key, probability_related FROM screen WHERE status='ok'"):
        prob = r['probability_related']
        if prob is None:
            continue
        if prob >= threshold:
            out['normal'] += 1
        else:
            rng = random.Random(f'{seed}:{r["pair_key"]}')
            out['audit' if rng.random() < audit_rate else 'screened_out'] += 1
    return out


def _collect(futs, label, total):
    """消费 futures：打进度；Ctrl+C 时取消未开始任务并立刻退出（已完成对已落库缓存）。"""
    done, next_mark = 0, 10
    try:
        for f in as_completed(futs):
            done += 1
            if total >= 20:
                pct = done * 100 // total
                if pct >= next_mark:
                    print(f'  {label} {done}/{total} ({pct}%)', flush=True)
                    next_mark = (pct // 10 + 1) * 10
            yield f.result()
    except KeyboardInterrupt:
        for f in futs:
            f.cancel()
        print(f'\n  interrupted: {label} 已完成 {done}/{total}（已完成对已缓存，重跑即续跑）', flush=True)
        raise SystemExit(130)


class Pipeline:
    """两阶段流水线：Jev Noul 初筛 → 路由（阈值/抽查）→ LLM 精判，分层缓存 + 断点恢复。

    网络调用并发执行；DB 操作用单连接 + 锁串行化（写很小，不构成瓶颈）。
    """

    def __init__(self, store, screen, detail, rubric):
        self.store = store
        self.screen = screen
        self.detail = detail
        self.rubric = rubric
        self.screen_rh = screen_rule_hash()
        self.screen_ch = screen_config_hash(screen)
        self.rubric_h = detail_rubric_hash(rubric)
        self.detail_ch = detail_config_hash(detail)
        self.screen_prompt_version = getattr(screen, 'prompt_version', 'screen-v1')
        self.detail_prompt_version = rubric.get('version', '')
        self.threshold = float(os.getenv('SCREEN_THRESHOLD', '0.5'))
        self.audit_rate = float(os.getenv('SCREEN_AUDIT_RATE', '0.0'))
        self.seed = int(os.getenv('RANDOM_SEED', '42'))
        self._lock = threading.Lock()

    def route_for(self, pair_key, prob):
        """给定初筛概率返回路由：normal / audit / screened_out / None(信息不足或失败)。"""
        if prob is None:
            return None
        if prob >= self.threshold:
            return 'normal'
        rng = random.Random(f'{self.seed}:{pair_key}')
        return 'audit' if rng.random() < self.audit_rate else 'screened_out'

    def _screen_pair(self, pair, by_id, hashes):
        pk = pair['pair_key']
        a_hash, b_hash = hashes.get(pair['a']), hashes.get(pair['b'])
        if a_hash is None or b_hash is None:
            return 'missing'
        with self._lock:
            if self.store.screen_hit(pk, a_hash, b_hash, self.screen_rh, self.screen_ch):
                return 'reused'
            a, b = by_id[pair['a']], by_id[pair['b']]
        record = dict(pair_key=pk, a_hash=a_hash, b_hash=b_hash, status=None, probability_related=None,
                      model=getattr(self.screen, 'model', ''), prompt_version=self.screen_prompt_version,
                      rule_hash=self.screen_rh, config_hash=self.screen_ch, usage=None, latency_ms=None,
                      raw_response=None, error=None, created_at=datetime.now(timezone.utc).isoformat())
        if not (_sufficient(a) and _sufficient(b)):
            record['status'] = 'insufficient_input'
        else:
            start = time.monotonic()
            try:
                res = self.screen.screen(a, b)
                record['status'] = 'ok'
                record['probability_related'] = res['probability_related']
                record['model'] = res.get('model', record['model'])
                record['usage'] = json.dumps(res.get('usage')) if res.get('usage') else None
                record['raw_response'] = json.dumps(res.get('raw')) if res.get('raw') else None
            except Exception:
                record['status'] = 'error'
                record['error'] = 'provider failure or invalid response'
            record['latency_ms'] = round((time.monotonic() - start) * 1000)
        with self._lock:
            self.store.put_screen(record)
        return record['status']

    def _detail_pair(self, pair, route, by_id, hashes):
        pk = pair['pair_key']
        a_hash, b_hash = hashes.get(pair['a']), hashes.get(pair['b'])
        with self._lock:
            if self.store.detail_hit(pk, a_hash, b_hash, self.rubric_h, self.detail_ch):
                return 'reused'
            a, b = by_id[pair['a']], by_id[pair['b']]
        record = dict(pair_key=pk, a_hash=a_hash, b_hash=b_hash, status=None, route=route, result=None,
                      label=None, model=getattr(self.detail, 'model', ''), prompt_version=self.detail_prompt_version,
                      rubric_hash=self.rubric_h, config_hash=self.detail_ch, usage=None, latency_ms=None,
                      raw_response=None, error=None, created_at=datetime.now(timezone.utc).isoformat())
        start = time.monotonic()
        try:
            result, raw = self.detail.judge(a, b)
            record['status'] = 'ok'
            record['result'] = json.dumps(result, ensure_ascii=False)
            record['label'] = label(result)
            if isinstance(raw, dict):
                record['model'] = raw.get('model', record['model'])
                record['usage'] = json.dumps(raw.get('usage')) if raw.get('usage') else None
                record['raw_response'] = json.dumps(raw)
        except Exception:
            record['status'] = 'error'
            record['error'] = 'provider failure or invalid response'
        record['latency_ms'] = round((time.monotonic() - start) * 1000)
        with self._lock:
            self.store.put_detail(record)
        return record['status']

    def run(self, limit=None, llm_cap=None, resume=None):
        pairs = sorted(self.store.distinct_pairs(), key=lambda x: x['pair_key'])
        if limit is not None:
            pairs = pairs[:limit]
        by_id = {p['spu_id']: p for p in self.store.active_products()}
        hashes = self.store.all_hashes()
        run_id = resume or (datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6])
        counts = dict(screen_called=0, screen_reused=0, screen_error=0, screen_insufficient=0,
                      detail_called=0, detail_reused=0, detail_error=0, detail_ok=0,
                      normal=0, audit=0, screened_out=0)
        if not resume:
            self.store.create_run(run_id, digest([self.screen_ch, self.detail_ch, self.screen_rh, self.rubric_h]),
                                  self.screen_rh, self.rubric_h)

        s_con = max(1, int(os.getenv('SCREEN_CONCURRENCY', '4')))
        total = len(pairs)
        print(f'  screen: {total} pairs (Jev Noul)...', flush=True)
        ex = ThreadPoolExecutor(max_workers=s_con)
        try:
            futs = [ex.submit(self._screen_pair, p, by_id, hashes) for p in pairs]
            for res in _collect(futs, 'screen', total):
                if res == 'reused':
                    counts['screen_reused'] += 1
                elif res == 'error':
                    counts['screen_error'] += 1
                elif res == 'insufficient_input':
                    counts['screen_insufficient'] += 1
                else:
                    counts['screen_called'] += 1
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        detail_pairs = []
        for p in pairs:
            sc = self.store.get_screen(p['pair_key'])
            if sc is None:
                continue
            prob = sc['probability_related'] if sc['status'] == 'ok' else None
            route = self.route_for(p['pair_key'], prob)
            if route in ('normal', 'audit'):
                detail_pairs.append((p, route))
                counts[route] += 1
            elif route == 'screened_out':
                counts['screened_out'] += 1
        if llm_cap is not None:
            detail_pairs = detail_pairs[:llm_cap]

        d_con = max(1, int(os.getenv('DETAIL_CONCURRENCY', '4')))
        total_d = len(detail_pairs)
        print(f'  detail: {total_d} pairs (LLM)...', flush=True)
        ex = ThreadPoolExecutor(max_workers=d_con)
        try:
            futs = [ex.submit(self._detail_pair, p, route, by_id, hashes)
                    for p, route in detail_pairs]
            for res in _collect(futs, 'detail', total_d):
                if res == 'reused':
                    counts['detail_reused'] += 1
                elif res == 'ok':
                    counts['detail_ok'] += 1
                    counts['detail_called'] += 1
                else:
                    counts['detail_error'] += 1
                    counts['detail_called'] += 1
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        self.store.set_run_status(run_id, 'complete', counts)
        return counts, run_id