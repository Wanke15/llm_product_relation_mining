from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol

from .core import DIMENSIONS, empty_dimension, model_product, validate_result


class Judge(Protocol):
    def judge(self, a: dict, b: dict) -> tuple[dict, dict]: ...


def post_json(url, key, payload):
    attempts = int(os.getenv('MAX_RETRIES', '3')) + 1
    for attempt in range(attempts):
        request = urllib.request.Request(url, json.dumps(payload).encode(),
            {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=float(os.getenv('REQUEST_TIMEOUT', '90'))) as res:
                return json.load(res)
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504, 529) or attempt == attempts - 1:
                # Do not persist response bodies: gateways can echo sensitive headers.
                raise RuntimeError(f'Provider HTTP {exc.code}') from None
            retry = exc.headers.get('Retry-After', '')
            delay = min(float(retry), 60) if retry.isdigit() else min(2 ** attempt, 30)
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise RuntimeError('Provider network/timeout failure') from None
            delay = min(2 ** attempt, 30)
        time.sleep(delay)
    raise RuntimeError('Invalid retry configuration')


class DemoJudge:
    model = 'demo-rules-not-a-model'

    def judge(self, a, b):
        same = bool(a['subcategory_id'] and a['subcategory_id'] == b['subcategory_id'])
        result = {k: empty_dimension(None, '演示规则，仅验证流程；不是模型判断') for k in DIMENSIONS}
        result['similarity']['score'] = 2 if same else 0
        return result, {'model': self.model, 'demo': True}


class LLMJudge:
    def __init__(self, rubric):
        self.rubric = rubric
        self.model = os.getenv('LLM_MODEL', '')
        self.key = os.getenv('LLM_API_KEY', '')
        if not self.key or not self.model:
            raise ValueError('Set LLM_API_KEY and LLM_MODEL in .env')

    def judge(self, a, b):
        schema = {k: empty_dimension() for k in DIMENSIONS}
        system = ('你是商品关系标注员。输出且仅输出 JSON。规则：' + json.dumps(self.rubric, ensure_ascii=False)
                  + '\n输出结构：' + json.dumps(schema, ensure_ascii=False)
                  + '\nscore 必须为整数0..3或null（unknown）。reason为简短理由。'
                  'evidence 为数组，每项 {"product":"a或b","field":"输入字段名","value":原字段完整值}。'
                  '有关系时应提供具体证据。conditions、conflicts、missing_information 为字符串数组。')
        payload = {'model': self.model, 'messages': [{'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps({'a': model_product(a), 'b': model_product(b)}, ensure_ascii=False)}]}
        if os.getenv('LLM_RESPONSE_FORMAT', 'json_object') == 'json_object':
            payload['response_format'] = {'type': 'json_object'}
        raw = post_json(os.getenv('LLM_URL', 'https://api.openai.com/v1/chat/completions'), self.key, payload)
        content = raw['choices'][0]['message']['content']
        result = json.loads(content)
        return validate_result(result, model_product(a), model_product(b)), raw


class JevJudge:
    def __init__(self, rubric):
        self.rubric = rubric
        self.model = os.getenv('JEV_MODEL', 'jev-latest')
        self.key = os.getenv('JEV_API_KEY', '')
        if not self.key:
            raise ValueError('Set JEV_API_KEY in .env')

    def judge(self, a, b):
        # Choice preserves an explicit unknown option; it is not a numeric midpoint.
        questions = {k: {'type': 'choice', 'instructions': self.rubric['rules'] + '\n' + self.rubric['dimensions'][k],
                         'criteria': self.rubric['levels']} for k in DIMENSIONS}
        raw = post_json(os.getenv('JEV_URL', 'https://api.typesafe.ai/v1/systemone'), self.key,
            {'model': self.model, 'state': {'a': model_product(a), 'b': model_product(b)}, 'questions': questions})
        result = {}
        for k in DIMENSIONS:
            answer = raw['answers'][k]
            choice = answer['choice']
            probs = answer['probabilities']
            if answer.get('type') != 'choice' or choice not in self.rubric['levels']:
                raise ValueError('Invalid Jev choice')
            if set(probs) != set(self.rubric['levels']) or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in probs.values()) or abs(sum(probs.values()) - 1) > .02:
                raise ValueError('Invalid Jev distribution')
            confidence = answer['confidence']
            if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
                raise ValueError('Invalid Jev confidence')
            result[k] = empty_dimension(None if choice == 'unknown' else int(choice))
            result[k].update(probabilities=probs, confidence=confidence)
        return validate_result(result, model_product(a), model_product(b)), raw


REGISTRY = {'demo': DemoJudge, 'llm': LLMJudge, 'jev': JevJudge}


def create_judge(name, rubric):
    if name not in REGISTRY:
        raise ValueError('Unknown provider: ' + name)
    return REGISTRY[name]() if name == 'demo' else REGISTRY[name](rubric)
