import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from relations import batch as batch_mod
from relations.document import embedding_text, embedding_text_hash, product_hash
from relations.embedding import HashEmbedding, OpenAIEmbedding
from relations.gender import compatible, gender_of, load_gender_rules
from relations.recall import (FUNCTIONAL, SIMILAR, STYLE, functional_query_text,
                              generate_candidates, load_roles, style_query_text)
from relations.store import Store
from relations.vector_index import NumpyIndex


def P(**kw):
    base = dict(spu_id='1', name='上衣', branch='女装', category_group='上装组', category='T恤',
                subcategory='长袖', subcategory_id='x', features='条纹', style='休闲', theme='主题',
                attribute_text='棉', season='冬', available_colors=['红', '白'], price=99.0,
                image_url='https://img/x.jpg')
    base.update(kw)
    return base


class DocumentTests(unittest.TestCase):
    def test_hashes_deterministic(self):
        self.assertEqual(product_hash(P()), product_hash(P()))
        self.assertEqual(embedding_text_hash(P()), embedding_text_hash(P()))

    def test_product_hash_invalidated_on_field_change(self):
        self.assertNotEqual(product_hash(P()), product_hash(P(name='裤子')))
        self.assertNotEqual(product_hash(P()), product_hash(P(price=1.0)))  # 价格影响判定哈希
        self.assertEqual(product_hash(P()), product_hash(P(image_url='https://img/z.jpg')))  # 图片不影响

    def test_embedding_text_excludes_price_image_codes(self):
        t = embedding_text(P())
        self.assertIn('商品名', t)
        self.assertIn('可选颜色', t)
        self.assertNotIn('99', t)       # price 不进文本
        self.assertNotIn('http', t)     # 图片不进文本
        self.assertNotIn('subcategory_id', t)  # 内部编码不入模板
        # 仅改价格，文本不变；仅改名称，文本变
        self.assertEqual(embedding_text(P()), embedding_text(P(price=1.0)))
        self.assertNotEqual(embedding_text(P()), embedding_text(P(name='裤子')))


class EmbeddingTests(unittest.TestCase):
    def test_hash_embedding_deterministic(self):
        e = HashEmbedding(64)
        a = e.embed(['苹果手机', '苹果手机'])
        self.assertEqual(a[0], a[1])
        self.assertEqual(len(a[0]), 64)

    def test_openai_requires_config(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                OpenAIEmbedding(url=None, key=None, model=None)


class IndexTests(unittest.TestCase):
    def test_search_excludes_self_and_ranks(self):
        idx = NumpyIndex(2)
        idx.add(['a', 'b', 'c'], [[1, 0], [0.9, 0.1], [0, 1]])
        res = idx.search_batch(idx.vectors_of(['a']), 2, [{'a'}])
        ids = [r[0] for r in res[0]]
        self.assertNotIn('a', ids)
        self.assertEqual(ids[0], 'b')

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            idx = NumpyIndex(3)
            idx.add(['x', 'y'], [[1, 0, 0], [0, 1, 0]])
            idx.save(Path(d))
            loaded = NumpyIndex(3).load(Path(d))
            self.assertEqual(loaded.ids, ['x', 'y'])
            self.assertEqual(loaded.dim, 3)


class RecallTests(unittest.TestCase):
    ROLES = {'上装组': {'role': '上衣', 'functional': ['下装裤类'], 'style': ['女鞋']}}

    def test_channels_are_distinct(self):
        p = P()
        sim = embedding_text(p)
        func = functional_query_text(p, self.ROLES)
        style = style_query_text(p, self.ROLES)
        self.assertIsNotNone(func)
        self.assertIsNotNone(style)
        self.assertEqual(len({sim, func, style}), 3)

    def test_unmapped_group_disables_complement(self):
        p = P(category_group='食品')
        self.assertIsNone(functional_query_text(p, self.ROLES))
        self.assertIsNone(style_query_text(p, self.ROLES))

    def test_bounded_no_self_dedup_sources(self):
        prods = {'a': P(spu_id='a', category_group='上装组'),
                 'b': P(spu_id='b', category_group='下装裤类', name='牛仔裤'),
                 'c': P(spu_id='c', category_group='下装裤类', name='休闲裤'),
                 'd': P(spu_id='d', category_group='女鞋', name='运动鞋'),
                 'e': P(spu_id='e', category_group='女鞋', name='靴子')}
        emb = HashEmbedding(32)
        idx = NumpyIndex(32)
        idx.add(list(prods.keys()), emb.embed([embedding_text(p) for p in prods.values()]))
        rows, report = generate_candidates(prods, idx, emb.embed, self.ROLES,
                                           {SIMILAR: 10, FUNCTIONAL: 10, STYLE: 10}, 'v1', fetch_k=30)
        self.assertTrue(all(r['anchor_spu_id'] != r['candidate_spu_id'] for r in rows))
        per = defaultdict(set)
        for r in rows:
            per[r['anchor_spu_id']].add(r['candidate_spu_id'])
        self.assertTrue(all(len(v) <= 30 for v in per.values()))
        self.assertEqual(report['coverage'][SIMILAR], len(prods))
        # 同一 (无序对, 通道) 唯一；可有多通道来源
        keys = set((r['pair_key'], r['channel']) for r in rows)
        self.assertEqual(len(keys), len(rows))
        self.assertIn(FUNCTIONAL, {r['channel'] for r in rows})


class StoreTests(unittest.TestCase):
    def test_product_embedding_cache_hit_and_invalidation(self):
        with tempfile.TemporaryDirectory() as d, Store(Path(d) / 'r.db') as s:
            s.upsert_products([P(spu_id='1')])
            th = embedding_text_hash(P(spu_id='1'))
            key = f"product:1:{th}:m:v"
            s.put_embedding(key, 'product', th, 'm', 'v', 3, [1, 2, 3])
            self.assertTrue(s.embedding_hit(key, th))
            th2 = embedding_text_hash(P(spu_id='1', name='新名'))
            key2 = f"product:1:{th2}:m:v"
            self.assertFalse(s.embedding_hit(key2, th2))  # 文本变更后旧缓存不命中新 key
            # 模型变更 → 新 key → 不命中
            key3 = f"product:1:{th}:m2:v"
            self.assertFalse(s.embedding_hit(key3, th))

    def test_upsert_and_active(self):
        with tempfile.TemporaryDirectory() as d, Store(Path(d) / 'r.db') as s:
            s.upsert_products([P(spu_id='1'), P(spu_id='2', branch='后勤')])
            self.assertEqual(len(s.active_ids()), 1)
            self.assertEqual(s.stats()['products_total'], 2)


class StoreConfigTests(unittest.TestCase):
    def test_roles_config_loads(self):
        roles, ver = load_roles()
        self.assertIsInstance(ver, int)
        self.assertIn('女装|上装组', roles)
        self.assertIn('男装|上装组', roles)
        self.assertIn('functional', roles['女装|上装组'])


class GenderTests(unittest.TestCase):
    def test_gender_of(self):
        rules, _ = load_gender_rules()
        self.assertEqual(gender_of('男装', '上装组', rules), '男')
        self.assertEqual(gender_of('女装', '上装组', rules), '女')
        self.assertEqual(gender_of('美妆', '面部底妆', rules), '中性')

    def test_compatible(self):
        self.assertTrue(compatible('女', '女'))
        self.assertFalse(compatible('女', '男'))
        self.assertTrue(compatible('女', '中性'))
        self.assertTrue(compatible('中性', '男'))

    def test_complement_split_by_branch(self):
        roles, _ = load_roles()
        wf = functional_query_text(P(branch='女装', category_group='上装组'), roles)
        mf = functional_query_text(P(branch='男装', category_group='上装组'), roles)
        self.assertIsNotNone(wf)
        self.assertIsNotNone(mf)
        self.assertNotIn('下装组', wf)       # 女装上衣不再混男装下装
        self.assertNotIn('下装裤类', mf)     # 男装上衣不再混女装下装
        self.assertNotEqual(wf, mf)

    def test_gender_filter_drops_opposite(self):
        roles, _ = load_roles()
        prods = {
            'w': P(spu_id='w', branch='女装', category_group='上装组', name='女上衣', gender='女'),
            'm1': P(spu_id='m1', branch='男装', category_group='上装组', name='男上衣', gender='男'),
            'm2': P(spu_id='m2', branch='男装', category_group='下装组', name='男裤', gender='男'),
            'w2': P(spu_id='w2', branch='女装', category_group='下装裤类', name='女裤', gender='女'),
        }
        emb = HashEmbedding(32)
        idx = NumpyIndex(32)
        idx.add(list(prods.keys()), emb.embed([embedding_text(p) for p in prods.values()]))
        rows, report = generate_candidates(prods, idx, emb.embed, roles,
                                           {SIMILAR: 10, FUNCTIONAL: 10, STYLE: 10}, 'v1', fetch_k=30)
        for r in rows:
            if r['anchor_spu_id'] == 'w':
                self.assertEqual(prods[r['candidate_spu_id']]['gender'], '女')
        self.assertGreaterEqual(report['gender_dropped'], 0)


class BatchTests(unittest.TestCase):
    def test_derive_base(self):
        self.assertEqual(OpenAIEmbedding._derive_base('https://x/compatible-mode/v1/embeddings'),
                         'https://x/compatible-mode/v1')
        self.assertEqual(OpenAIEmbedding._derive_base('https://api.openai.com/v1/embeddings'),
                         'https://api.openai.com/v1')

    @patch('relations.embedding.post_json')
    def test_embed_sync_chunks(self, post):
        prov = OpenAIEmbedding(url='https://x/v1/embeddings', key='k', model='m', batch_size=2)
        post.side_effect = lambda url, key, payload: {'data': [{'embedding': [1.0]} for _ in payload['input']]}
        out = prov._embed_sync(['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(len(out), 5)
        self.assertEqual(post.call_count, 3)

    @patch('relations.embedding.batch_embed')
    def test_embed_routes_to_batch_then_sync(self, bm):
        bm.return_value = [[0.0]] * 8
        prov = OpenAIEmbedding(url='https://x/v1/embeddings', key='k', model='m', batch_min=5, batch_size=2)
        prov.embed(['t'] * 8)
        bm.assert_called_once()
        with patch('relations.embedding.post_json') as post:
            post.return_value = {'data': [{'embedding': [1.0]}, {'embedding': [1.0]}]}
            prov.embed(['t'] * 2)  # 低于阈值走同步
        bm.assert_called_once()

    def test_batch_embed_parse(self):
        completed = json.dumps({'status': 'completed', 'output_file_id': 'out-1'}).encode()
        output = (json.dumps({'custom_id': 'emb-0', 'response': {'status_code': 200,
                                                                 'body': {'data': [{'embedding': [1.0, 2.0]}]}}}) + '\n' +
                  json.dumps({'custom_id': 'emb-1', 'response': {'status_code': 200,
                                                                 'body': {'data': [{'embedding': [3.0, 4.0]}]}}}) + '\n')
        with patch.object(batch_mod, 'upload_file', return_value={'id': 'file-1'}), \
                patch.object(batch_mod, 'post_json', return_value={'id': 'batch-1'}), \
                patch.object(batch_mod, '_get', side_effect=[completed, output.encode()]), \
                patch.object(batch_mod.time, 'sleep'):
            out = batch_mod.batch_embed('https://x/v1', 'k', 'm', ['a', 'b'])
        self.assertEqual(out, [[1.0, 2.0], [3.0, 4.0]])


if __name__ == '__main__':
    unittest.main()