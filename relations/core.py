from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = ('similarity', 'functional_complement', 'style_pairing')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def clean(value):
    value = (value or '').strip()
    return None if value.lower() in ('', 'null', 'none', 'nan', 'n/a') else value


def load_products(path):
    products = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'prod_id', 'goods_name', 'catg_name', 'sub_catg_id'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('CSV missing columns: ' + ', '.join(sorted(required - set(reader.fieldnames or []))))
        for row in reader:
            pid = clean(row['prod_id'])
            if not pid or pid in products:
                raise ValueError(f'Missing or duplicate SPU: {pid}')
            p = {key: clean(row.get(source)) for key, source in {
                'spu_id': 'prod_id', 'name': 'goods_name', 'branch': 'branch_name',
                'category_group': 'catg_grp_name', 'category': 'catg_name',
                'subcategory': 'sub_catg_name', 'subcategory_id': 'sub_catg_id',
                'features': 'ftr', 'style': 'style', 'theme': 'subj',
                'attribute_text': 'mat', 'season': 'season', 'image_url': 'prim_img'
            }.items()}
            p['available_colors'] = [c.strip() for c in (row.get('color_total') or '').split('/') if c.strip()]
            p['price'] = float(row['saleprice']) if clean(row.get('saleprice')) else None
            if p['theme'] in ('0', '非主题'):
                p['theme'] = None
            products[pid] = p
    return products


def model_product(p):
    # Text-only v1: URLs are for review only, not evidence of having seen an image.
    return {k: v for k, v in p.items() if k != 'image_url'}


def candidates(products, limit=200, seed=42):
    """Bounded, reproducible stratified discovery; never materializes N squared."""
    rng = random.Random(seed)
    eligible = sorted(k for k, p in products.items() if p['branch'] != '后勤' and p['name'])
    if len(eligible) < 2:
        raise ValueError('Need at least two eligible products')
    groups = defaultdict(list)
    styles = defaultdict(list)
    for pid in eligible:
        p = products[pid]
        if p['subcategory_id']:
            groups[(p['branch'], p['subcategory_id'])].append(pid)
        if p['style']:
            styles[(p['branch'], p['style'])].append(pid)
    groups = [v for v in groups.values() if len(v) > 1]
    styles = [v for v in styles.values() if len(v) > 1]
    pairs = {}
    # Cross-category samples include potential complements; these are NOT labels.
    for i in range(max(1000, limit * 100)):
        if len(pairs) >= limit:
            break
        source = ('same_subcategory', 'same_style_cross_category', 'random')[i % 3]
        pool = rng.choice(groups) if source == 'same_subcategory' and groups else eligible
        if source == 'same_style_cross_category':
            pool = rng.choice(styles) if styles else eligible
        a, b = sorted(rng.sample(pool, 2))
        if source == 'same_style_cross_category' and products[a]['category'] == products[b]['category']:
            continue
        if (a, b) not in pairs:
            pairs[a, b] = {'pair_id': digest([a, b])[:20], 'a': a, 'b': b, 'source': source}
    return list(pairs.values())


def validate_result(result, a, b):
    if set(result) != set(DIMENSIONS):
        raise ValueError('Result must contain exactly the three relation dimensions')
    for name in DIMENSIONS:
        d = result[name]
        if not isinstance(d, dict):
            raise ValueError('Dimension must be an object')
        score = d.get('score')
        if score is not None and (type(score) is not int or score not in range(4)):
            raise ValueError('Score must be null or integer 0..3')
        if 'score' not in d:
            raise ValueError('Missing score')
        if d.get('reason') is not None and not isinstance(d['reason'], str):
            raise ValueError('Invalid reason')
        for field in ('conditions', 'conflicts', 'missing_information'):
            if not isinstance(d.get(field), list) or any(not isinstance(v, str) for v in d[field]):
                raise ValueError('Invalid ' + field)
        if not isinstance(d.get('evidence'), list):
            raise ValueError('Invalid evidence')
        for ev in d['evidence']:
            if not isinstance(ev, dict) or ev.get('product') not in ('a', 'b'):
                raise ValueError('Invalid evidence product')
            p = a if ev['product'] == 'a' else b
            if ev.get('field') not in p or ev.get('value') != p[ev['field']]:
                raise ValueError('Evidence not equal to input field value')
    return result


def empty_dimension(score=None, reason=None):
    return dict(score=score, reason=reason, evidence=[], conditions=[], conflicts=[], missing_information=[])


def label(result, threshold=2):
    sim = result['similarity']['score']
    comps = [result[k]['score'] for k in DIMENSIONS[1:]]
    s = sim is not None and sim >= threshold
    c = any(v is not None and v >= threshold for v in comps)
    if s and c:
        return 'both'
    if s:
        return 'similar'
    if c:
        return 'complement'
    return 'unknown' if any(v is None for v in [sim, *comps]) else 'none'
