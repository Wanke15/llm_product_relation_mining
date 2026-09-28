from __future__ import annotations

import os

from .core import digest, model_product
from .providers import post_json

SCREEN_PROMPT_VERSION = 'screen-v1'

# Noul 初筛：回答「是否存在至少一种值得进一步评估的关系」，输出 0(否)~1(是) 概率，不是关系强度。
SCREEN_INSTRUCTIONS = (
    '根据已有商品信息，两件商品是否存在至少一种值得进一步评估的关系：'
    '满足相近核心需求的替代关系、具体任务中的功能互补，或有属性依据的穿搭互补？'
    '仅同品牌、同宽泛场景或同泛风格不足以成立。这是候选初筛，不要求完整推荐理由。')
SCREEN_CRITERIA = {
    'true': '存在值得进一步评估的替代 / 功能互补 / 穿搭互补关系，且有属性依据',
    'false': '仅同品牌、同宽泛场景或同泛风格，缺乏属性依据，或无此类关系',
}


def screen_rule_hash():
    return digest({'version': SCREEN_PROMPT_VERSION, 'instructions': SCREEN_INSTRUCTIONS,
                   'criteria': SCREEN_CRITERIA})


def screen_config_hash(screen):
    """后端配置版本（端点 + 模型别名 + 实际 provider 模型），不含密钥。"""
    return digest([os.getenv('JEV_URL', 'https://api.typesafe.ai/v1/systemone'),
                   os.getenv('JEV_MODEL', 'jev-latest'),
                   getattr(screen, 'model', '')])


class JevScreen:
    """Jev Noul 初筛：一个 Noul 问题，返回关系成立概率。"""
    prompt_version = SCREEN_PROMPT_VERSION

    def __init__(self):
        self.model = os.getenv('JEV_MODEL', 'jev-latest')
        self.key = os.getenv('JEV_API_KEY', '')
        if not self.key:
            raise ValueError('Set JEV_API_KEY in .env')

    def screen(self, a, b):
        questions = {'related': {'type': 'noul', 'instructions': SCREEN_INSTRUCTIONS,
                                 'criteria': SCREEN_CRITERIA}}
        raw = post_json(os.getenv('JEV_URL', 'https://api.typesafe.ai/v1/systemone'), self.key,
                        {'model': self.model,
                         'state': {'a': model_product(a), 'b': model_product(b)},
                         'questions': questions})
        answer = raw['answers']['related']
        if answer.get('type') != 'noul' or not isinstance(answer.get('noul'), (int, float)):
            raise ValueError('Invalid Noul response')
        prob = float(answer['noul'])
        if not 0 <= prob <= 1:
            raise ValueError('Noul out of range')
        return {'probability_related': prob, 'model': raw.get('model', self.model),
                'usage': raw.get('usage'), 'raw': raw}


class DemoScreen:
    """离线初筛：规则占位（同细类视为可能替代），仅验证流程，非模型判断。"""
    prompt_version = SCREEN_PROMPT_VERSION

    def screen(self, a, b):
        if a.get('subcategory_id') and a.get('subcategory_id') == b.get('subcategory_id'):
            p = 0.85
        elif a.get('category') and a.get('category') == b.get('category'):
            p = 0.5
        else:
            p = 0.2
        return {'probability_related': p, 'model': 'demo-screen-not-a-model', 'usage': None, 'raw': None}


def create_screen():
    return JevScreen() if os.getenv('JEV_API_KEY') else DemoScreen()