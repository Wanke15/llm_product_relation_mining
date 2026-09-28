from __future__ import annotations

import json
from pathlib import Path

NEUTRAL = '中性'


def default_path():
    return Path(__file__).resolve().parent / 'config' / 'gender_rules.json'


def load_gender_rules(path=None):
    data = json.loads(Path(path or default_path()).read_text(encoding='utf-8'))
    return data['gender'], data.get('version')


def gender_of(branch, category_group, rules):
    """按 部门|类目组 查性别；未命中默认中性。"""
    return rules.get(f"{branch}|{category_group}") or NEUTRAL


def compatible(ga, gb):
    """召回兼容性：只丢弃男女相反；中性锚点、中性候选或同性别均保留。"""
    return ga == NEUTRAL or gb == NEUTRAL or ga == gb