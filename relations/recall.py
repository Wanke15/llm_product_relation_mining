from __future__ import annotations

import json
from pathlib import Path

from .core import digest
from .gender import compatible

SIMILAR = 'similar'
FUNCTIONAL = 'functional'
STYLE = 'style'
CHANNELS = (SIMILAR, FUNCTIONAL, STYLE)

QUERY_TPL_VERSION = 'recall-query-v2'


def load_roles(path=None):
    path = Path(path or (Path(__file__).resolve().parent / 'config' / 'complement_roles.json'))
    data = json.loads(path.read_text(encoding='utf-8'))
    return data['roles'], data.get('version')


_SEASON = {'春': '春季', '夏': '夏季', '秋': '秋季', '冬': '冬季',
           '春1': '春季', '夏1': '夏季', '夏2': '夏季', '秋1': '秋季', '冬1': '冬季', '冬2': '冬季'}


def _season_label(season):
    return _SEASON.get(season or '', season)


def _style_season(p):
    """取锚点的风格/季节作为方向性上下文（不掺角色名/商品名）。"""
    parts = []
    if p.get('style'):
        parts.append('风格' + p['style'])
    s = p.get('season')
    if s and s != '四季':
        parts.append('季节' + _season_label(s))
    return parts


def _role_for(p, roles):
    return roles.get(f"{p.get('branch')}|{p.get('category_group')}") or roles.get(p.get('category_group'))


def functional_query_text(p, roles):
    """功能互补：只描述目标类目 + 风格/季节，不带自身角色，避免把同方向同类拉回来。"""
    r = _role_for(p, roles)
    if not r or not r.get('functional'):
        return None
    return '，'.join(['、'.join(r['functional'])] + _style_season(p))


def style_query_text(p, roles):
    """风格搭配：目标类目 + 风格/季节（风格协调本就依赖锚点风格），同样不带角色名。"""
    r = _role_for(p, roles)
    if not r or not r.get('style'):
        return None
    return '，'.join(['、'.join(r['style'])] + _style_season(p))


def _allocate(quotas, enabled):
    """把停用通道的配额均摊给启用通道，总额不变。"""
    cap = {c: quotas[c] for c in enabled}
    absent = sum(quotas[c] for c in CHANNELS if c not in enabled)
    if absent and enabled:
        share, rem = divmod(absent, len(enabled))
        for j, c in enumerate(enabled):
            cap[c] += share + (1 if j < rem else 0)
    return cap


def _channel_hits(anchors, by_id, roles, channel, embed, index, fetch_k):
    """返回与 anchors 对齐的逐锚命中列表；停用的锚为 []。"""
    results = [[] for _ in anchors]
    pairs = []  # (anchor_index, text)
    builder = functional_query_text if channel == FUNCTIONAL else style_query_text
    for i, a in enumerate(anchors):
        text = builder(by_id[a], roles)
        if text is not None:
            pairs.append((i, text))
    if not pairs:
        return results
    # 去重后批量 embedding（embed 内部已做缓存）
    uniq, order = {}, []
    for _, text in pairs:
        if text not in uniq:
            uniq[text] = len(order)
            order.append(text)
    tv = dict(zip(order, embed(order)))
    qvec = [tv[text] for _, text in pairs]
    excludes = [{anchors[i]} for i, _ in pairs]
    res = index.search_batch(qvec, fetch_k, excludes)
    for j, (i, _) in enumerate(pairs):
        results[i] = res[j]
    return results


def generate_candidates(by_id, index, embed, roles, quotas, retrieval_version, fetch_k=30, anchors=None):
    """三通道召回 + 无序去重 + 来源合并，返回 (rows, report)。

    by_id: {spu_id: 商品}；anchors 默认取 index.ids（活跃且已索引的商品），可传子集限量。
    rows 每项: {pair_key, anchor_spu_id, candidate_spu_id, channel, rank, retrieval_score, retrieval_version}
    同一无序商品对可来自多通道，各自成行保留来源。
    """
    anchors = list(anchors) if anchors is not None else list(index.ids)
    sim = index.search_batch(index.vectors_of(anchors), fetch_k, [{a} for a in anchors])
    func = _channel_hits(anchors, by_id, roles, FUNCTIONAL, embed, index, fetch_k)
    style = _channel_hits(anchors, by_id, roles, STYLE, embed, index, fetch_k)

    rows = []
    seen_key = set()  # (pair_key, channel)：同一无序对经同一通道被两端分别召回时，合并为一条来源
    gender_dropped = 0
    coverage = {SIMILAR: len(anchors), FUNCTIONAL: 0, STYLE: 0}
    for i, a in enumerate(anchors):
        hits = [(SIMILAR, sim[i])]
        if func[i]:
            hits.append((FUNCTIONAL, func[i]))
            coverage[FUNCTIONAL] += 1
        if style[i]:
            hits.append((STYLE, style[i]))
            coverage[STYLE] += 1
        caps = _allocate(quotas, [c for c, _ in hits])
        ga = by_id[a].get('gender', '中性')
        sources = {}  # candidate -> list[(channel, rank, score)]
        for channel, h in hits:
            for rank, (cid, score) in enumerate(h):
                if rank >= caps[channel]:
                    break
                if not compatible(ga, by_id.get(cid, {}).get('gender', '中性')):
                    gender_dropped += 1
                    continue
                sources.setdefault(cid, []).append((channel, rank, score))
        for cid, srcs in sources.items():
            pair_key = digest(sorted((a, cid)))
            for channel, rank, score in srcs:
                key = (pair_key, channel)
                if key in seen_key:
                    continue
                seen_key.add(key)
                rows.append(dict(pair_key=pair_key, anchor_spu_id=a, candidate_spu_id=cid,
                                 channel=channel, rank=rank, retrieval_score=score,
                                 retrieval_version=retrieval_version))
    report = dict(anchors=len(anchors), pairs=len({r['pair_key'] for r in rows}),
                  rows=len(rows), coverage=coverage, gender_dropped=gender_dropped)
    return rows, report