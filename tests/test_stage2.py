import json
import tempfile
import unittest
from pathlib import Path

from relations.core import DIMENSIONS, empty_dimension
from relations.pipeline import Pipeline, preview_threshold
from relations.store import Store


def prod(**kw):
    base = dict(spu_id='1', name='上衣', branch='女装', category_group='上装组', category='T恤',
                subcategory='长袖', subcategory_id='x', features='条纹', style='休闲', theme='主题',
                attribute_text='棉', season='冬', available_colors=['红'], price=99.0, image_url='https://x/y.jpg')
    base.update(kw)
    return base


class FakeScreen:
    prompt_version = 'screen-v1'
    model = 'fake-screen'

    def __init__(self, prob=0.8, fail=False):
        self.prob, self.fail, self.calls = prob, fail, 0

    def screen(self, a, b):
        self.calls += 1
        if self.fail:
            raise RuntimeError('boom')
        return {'probability_related': self.prob, 'model': self.model,
                'usage': {'input_tokens': 1, 'output_tokens': 1},
                'raw': {'answers': {'related': {'type': 'noul', 'noul': self.prob}}}}


class FakeDetail:
    model = 'fake-detail'

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def judge(self, a, b):
        self.calls += 1
        if self.fail:
            raise RuntimeError('boom')
        result = {k: empty_dimension(0, '规则占位') for k in DIMENSIONS}
        return result, {'model': self.model, 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}


RUBRIC = {'version': 'spu-relations-v1', 'rules': '', 'dimensions': {}, 'levels': {}}


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / 'r.db')
        self.store.upsert_products([prod(spu_id='a'), prod(spu_id='b')])
        self.by_id = {p['spu_id']: p for p in self.store.active_products()}
        self.hashes = self.store.all_hashes()
        self.pair = {'pair_key': 'pk1', 'a': 'a', 'b': 'b'}

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _pipe(self, screen=None, detail=None):
        return Pipeline(self.store, screen or FakeScreen(), detail or FakeDetail(), RUBRIC)

    def test_screen_cache_reuse_and_product_invalidation(self):
        screen = FakeScreen(0.8)
        pipe = self._pipe(screen=screen)
        self.assertEqual(pipe._screen_pair(self.pair, self.by_id, self.hashes), 'ok')
        self.assertEqual(screen.calls, 1)
        self.assertEqual(pipe._screen_pair(self.pair, self.by_id, self.hashes), 'reused')
        self.assertEqual(screen.calls, 1)
        # 商品变更 → product_hash 变 → 缓存失效 → 重跑
        self.store.upsert_products([prod(spu_id='a', name='新名字')])
        self.assertEqual(pipe._screen_pair(self.pair, self.by_id, self.store.all_hashes()), 'ok')
        self.assertEqual(screen.calls, 2)

    def test_screen_error_not_low_prob(self):
        pipe = self._pipe(screen=FakeScreen(fail=True))
        pipe._screen_pair(self.pair, self.by_id, self.hashes)
        sc = self.store.get_screen('pk1')
        self.assertEqual(sc['status'], 'error')
        self.assertIsNone(sc['probability_related'])

    def test_insufficient_input_no_fake_prob(self):
        self.store.upsert_products([prod(spu_id='x', name='某商品', category=None, category_group=None)])
        by_id = {p['spu_id']: p for p in self.store.active_products()}
        hashes = self.store.all_hashes()
        pipe = self._pipe(screen=FakeScreen())
        pipe._screen_pair({'pair_key': 'pk2', 'a': 'a', 'b': 'x'}, by_id, hashes)
        sc = self.store.get_screen('pk2')
        self.assertEqual(sc['status'], 'insufficient_input')
        self.assertIsNone(sc['probability_related'])

    def test_threshold_change_no_rescreen(self):
        screen = FakeScreen(0.6)
        pipe = self._pipe(screen=screen)
        pipe._screen_pair(self.pair, self.by_id, self.hashes)
        self.assertEqual(pipe.route_for('pk1', 0.6), 'normal')  # 阈值 0.5
        pipe.threshold = 0.8  # 改阈值：只重算路由，不重跑 Jev
        self.assertEqual(pipe.route_for('pk1', 0.6), 'screened_out')
        self.assertEqual(screen.calls, 1)

    def test_audit_sampling_reproducible(self):
        pipe = self._pipe()
        pipe.threshold, pipe.audit_rate = 0.9, 0.5
        self.assertEqual(pipe.route_for('pk1', 0.2), pipe.route_for('pk1', 0.2))
        routes = [pipe.route_for(f'pk{i}', 0.2) for i in range(1000)]
        audit = sum(1 for r in routes if r == 'audit')
        self.assertTrue(400 < audit < 600, audit)

    def test_below_threshold_not_routed_to_llm(self):
        pipe = self._pipe()
        pipe.threshold, pipe.audit_rate = 0.9, 0.0
        self.assertEqual(pipe.route_for('pk1', 0.3), 'screened_out')
        self.assertEqual(pipe.route_for('pk1', 0.95), 'normal')

    def test_detail_cache_and_reuse(self):
        detail = FakeDetail()
        pipe = self._pipe(detail=detail)
        pipe._screen_pair(self.pair, self.by_id, self.hashes)  # prob 0.8 >= 0.5
        self.assertEqual(pipe._detail_pair(self.pair, 'normal', self.by_id, self.hashes), 'ok')
        self.assertEqual(detail.calls, 1)
        self.assertEqual(pipe._detail_pair(self.pair, 'normal', self.by_id, self.hashes), 'reused')
        self.assertEqual(detail.calls, 1)

    def test_detail_result_persisted(self):
        pipe = self._pipe()
        pipe._screen_pair(self.pair, self.by_id, self.hashes)
        pipe._detail_pair(self.pair, 'normal', self.by_id, self.hashes)
        d = self.store.get_detail('pk1')
        self.assertEqual(d['status'], 'ok')
        self.assertEqual(d['route'], 'normal')
        self.assertIn('similarity', json.loads(d['result']))

    def test_preview_threshold_counts(self):
        pipe = self._pipe()
        self.store.put_screen(dict(pair_key='s1', a_hash='h', b_hash='h', status='ok', probability_related=0.9,
                                   model='m', prompt_version='v', rule_hash='r', config_hash='c',
                                   usage=None, latency_ms=0, raw_response=None, error=None,
                                   created_at='now'))
        self.store.put_screen(dict(pair_key='s2', a_hash='h', b_hash='h', status='ok', probability_related=0.1,
                                   model='m', prompt_version='v', rule_hash='r', config_hash='c',
                                   usage=None, latency_ms=0, raw_response=None, error=None,
                                   created_at='now'))
        out = preview_threshold(self.store, 0.5, 0.0, 42)
        self.assertEqual(out, {'normal': 1, 'audit': 0, 'screened_out': 1})

    def test_full_two_stage_run_skips_screen_on_resume(self):
        screen, detail = FakeScreen(0.8), FakeDetail()
        pipe = self._pipe(screen=screen, detail=detail)
        # 造候选表 + 刷新无序对表（对应 retrieve 的行为）
        self.store.replace_candidates([('pk1', 'similar', 'a', 'b', 0, 0.5, 'v')])
        self.store.refresh_pairs()
        counts, run_id = pipe.run(llm_cap=100)
        self.assertEqual(screen.calls, 1)
        self.assertEqual(detail.calls, 1)
        self.assertEqual(counts['detail_ok'], 1)
        counts2, _ = pipe.run(resume=run_id)  # 恢复：一阶段不重跑
        self.assertEqual(screen.calls, 1)  # 未重新初筛
        self.assertEqual(counts2['screen_reused'], 1)
        self.assertEqual(detail.calls, 1)  # 未重新精判


if __name__ == '__main__':
    unittest.main()