import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from relations.core import ROOT, DIMENSIONS, candidates, empty_dimension, label, load_products, validate_result
from relations.providers import JevJudge, LLMJudge, post_json


class RelationsTests(unittest.TestCase):
    def setUp(self):
        self.rubric = json.loads((ROOT / 'prompts/rubric.json').read_text(encoding='utf-8'))
        self.a = dict(spu_id='1', name='上衣', branch='女装', subcategory_id='a', category='上衣', style='休闲')
        self.b = dict(self.a, spu_id='2')

    def test_candidates_reproducible_unique_no_self(self):
        products = {str(i): dict(self.a, spu_id=str(i), subcategory_id=str(i % 3), category=str(i % 3)) for i in range(30)}
        pairs = candidates(products, 25)
        self.assertEqual(pairs, candidates(products, 25))
        self.assertEqual(len(pairs), 25)
        self.assertEqual(len({p['pair_id'] for p in pairs}), 25)
        self.assertTrue(all(p['a'] != p['b'] for p in pairs))

    def test_unknown_is_not_none(self):
        result = {k: empty_dimension() for k in DIMENSIONS}
        self.assertEqual(label(result), 'unknown')
        for d in result.values():
            d['score'] = 0
        self.assertEqual(label(result), 'none')
        result['similarity']['score'] = 2
        result['style_pairing']['score'] = 2
        self.assertEqual(label(result), 'both')

    def test_fabricated_evidence_and_boolean_scores_rejected(self):
        result = {k: empty_dimension(0) for k in DIMENSIONS}
        result['similarity']['evidence'] = [dict(product='a', field='style', value='淑女')]
        with self.assertRaises(ValueError):
            validate_result(result, self.a, self.b)
        result['similarity']['evidence'] = []
        result['similarity']['score'] = True
        with self.assertRaises(ValueError):
            validate_result(result, self.a, self.b)

    @patch.dict(os.environ, {'JEV_API_KEY': 'test', 'JEV_MODEL': 'jev-latest'})
    @patch('relations.providers.post_json')
    def test_jev_native_contract_preserves_unknown(self, post):
        answer = dict(type='choice', choice='unknown', probabilities={'0': 0, '1': 0, '2': 0, '3': 0, 'unknown': 1}, confidence=1)
        post.return_value = dict(answers={k: answer for k in DIMENSIONS}, model='jev-test')
        result, raw = JevJudge(self.rubric).judge(self.a, self.b)
        self.assertIsNone(result['similarity']['score'])
        self.assertIsNone(result['similarity']['reason'])
        payload = post.call_args.args[2]
        self.assertEqual(set(payload['questions']), set(DIMENSIONS))
        self.assertEqual(payload['questions']['similarity']['type'], 'choice')

    @patch.dict(os.environ, {'LLM_API_KEY': 'test', 'LLM_MODEL': 'test'})
    @patch('relations.providers.post_json')
    def test_llm_contract(self, post):
        result = {k: empty_dimension(0, '用途不同') for k in DIMENSIONS}
        post.return_value = {'choices': [{'message': {'content': json.dumps(result)}}]}
        parsed, _ = LLMJudge(self.rubric).judge(self.a, self.b)
        self.assertEqual(parsed, result)
        self.assertIn('messages', post.call_args.args[2])

    def test_duplicate_spu_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'p.csv'
            path.write_text('prod_id,goods_name,catg_name,sub_catg_id\n1,A,C,2\n1,B,C,2\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_products(path)

    @patch.dict(os.environ, {'MAX_RETRIES': '2'})
    @patch('relations.providers.time.sleep')
    @patch('relations.providers.urllib.request.urlopen')
    def test_retry_rate_limit_but_not_auth(self, opening, sleep):
        opening.side_effect = urllib.error.HTTPError('https://test', 429, 'rate', {}, None)
        with self.assertRaisesRegex(RuntimeError, 'HTTP 429'):
            post_json('https://test', 'secret', {})
        self.assertEqual(opening.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        opening.reset_mock()
        opening.side_effect = urllib.error.HTTPError('https://test', 401, 'secret', {}, None)
        with self.assertRaisesRegex(RuntimeError, '^Provider HTTP 401$'):
            post_json('https://test', 'secret', {})
        self.assertEqual(opening.call_count, 1)


if __name__ == '__main__':
    unittest.main()
