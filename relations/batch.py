from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid

from .providers import post_json

_RETRYABLE = (408, 429, 500, 502, 503, 504, 529)


def _attempts():
    return int(os.getenv('MAX_RETRIES', '3')) + 1


def _delay(retry_after, attempt):
    head = (retry_after or '').strip()
    return min(float(head), 60) if head.isdigit() else min(2 ** attempt, 30)


def _with_retry(fn):
    for attempt in range(_attempts()):
        try:
            return fn()
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE or attempt == _attempts() - 1:
                raise RuntimeError(f'Batch API HTTP {exc.code}') from None
            delay = _delay(exc.headers.get('Retry-After', ''), attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == _attempts() - 1:
                raise RuntimeError('Batch API network/timeout failure') from None
            delay = min(2 ** attempt, 30)
        time.sleep(delay)
    raise RuntimeError('Invalid retry configuration')


def _get(url, key, timeout=180):
    def call():
        req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.read()
    return _with_retry(call)


def upload_file(base_url, key, jsonl_bytes, filename='batch.jsonl'):
    boundary = '----relbatch' + uuid.uuid4().hex
    body = b''
    body += (f'--{boundary}\r\n'
             f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             'Content-Type: application/octet-stream\r\n\r\n').encode('utf-8')
    body += jsonl_bytes + b'\r\n'
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n').encode('utf-8')
    body += f'--{boundary}--\r\n'.encode('utf-8')

    def call():
        req = urllib.request.Request(base_url.rstrip('/') + '/files', data=body,
                                     headers={'Authorization': 'Bearer ' + key,
                                              'Content-Type': 'multipart/form-data; boundary=' + boundary})
        with urllib.request.urlopen(req, timeout=180) as res:
            return json.load(res)
    return _with_retry(call)


def batch_embed(base_url, key, model, texts, completion_window='24h',
                poll_interval=5, max_wait=86400, sync_fn=None):
    """OpenAI 兼容 Batch（文件输入）走 embedding：上传 JSONL → 建任务 → 轮询 → 下载。

    失败的行可选走 sync_fn 兜底（小批量重试）；全部失败则抛错，不静默返回残缺向量。
    """
    base = base_url.rstrip('/')
    lines = [json.dumps({'custom_id': f'emb-{i}', 'method': 'POST', 'url': '/v1/embeddings',
                         'body': {'model': model, 'input': t, 'encoding_format': 'float'}},
                        ensure_ascii=False)
             for i, t in enumerate(texts)]
    content = ('\n'.join(lines) + '\n').encode('utf-8')
    file_id = upload_file(base, key, content)['id']
    batch = post_json(base + '/batches', key,
                      {'input_file_id': file_id, 'endpoint': '/v1/embeddings',
                       'completion_window': completion_window})
    batch_id = batch['id']
    status, output_id = None, None
    waited = 0
    last_status = None
    while waited < max_wait:
        st = json.loads(_get(base + '/batches/' + batch_id, key))
        status = st.get('status')
        output_id = st.get('output_file_id')
        if status != last_status:  # 只在状态变化时打点，不按轮询刷屏
            print(f'  batch {batch_id}: {status}', flush=True)
            last_status = status
        if status in ('completed', 'failed', 'expired', 'cancelled'):
            break
        time.sleep(poll_interval)
        waited += poll_interval
    if status != 'completed':
        raise RuntimeError(f'Batch embedding job status={status}')
    output_text = _get(base + '/files/' + output_id + '/content', key).decode('utf-8')
    vectors = [None] * len(texts)
    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cid = obj.get('custom_id') or ''
        if not cid.startswith('emb-'):
            continue
        idx = int(cid.split('-')[-1])
        resp = obj.get('response') or {}
        if resp.get('status_code') not in (200, None) or obj.get('error'):
            continue
        data = ((resp.get('body') or {}).get('data')) or []
        if data and isinstance(data[0].get('embedding'), list):
            vectors[idx] = [float(x) for x in data[0]['embedding']]
    missing = [i for i, v in enumerate(vectors) if v is None]
    if missing and sync_fn:
        for i, v in zip(missing, sync_fn([texts[i] for i in missing])):
            vectors[i] = v
    missing = [i for i, v in enumerate(vectors) if v is None]
    if missing:
        raise RuntimeError(f'Batch embedding: {len(missing)}/{len(texts)} requests failed')
    return vectors