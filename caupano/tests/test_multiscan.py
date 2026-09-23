"""CPU split regression checks; no raw data, model or CUDA required."""
import unittest

from qwen_pano.caupano.tools.multiscan import partition


class MultiScanTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(sample_id=f'{scan}_{i}', scan_id=scan, split='train')
                     for scan in ['a', 'b', 'c'] for i in range(15)]

    def test_heldout_building_never_enters_fit(self):
        fit, dev = partition(self.rows, ['a','b'], 'c', 2, 4, 42)
        self.assertEqual(len(fit), 26)
        self.assertEqual({r['scan_id'] for r in fit}, {'a','b'})
        self.assertFalse({r['sample_id'] for r in fit} & {r['sample_id'] for r in dev})
        self.assertEqual(sum(r['evaluation_group']=='same_building' for r in dev), 4)
        self.assertEqual(sum(r['evaluation_group']=='heldout_building' for r in dev), 4)
        self.assertEqual({r['evaluation_group'] for r in dev[:2]}, {'same_building','heldout_building'})
        # File order must not silently change frozen dev selection.
        self.assertEqual((fit,dev), partition(self.rows[::-1], ['a','b'], 'c', 2, 4, 42))

    def test_reject_source_test_and_duplicates(self):
        bad = [dict(r) for r in self.rows]
        bad[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            partition(bad, ['a','b'], 'c', 2, 4, 42)
        with self.assertRaises(ValueError):
            partition(self.rows+[self.rows[0]], ['a','b'], 'c', 2, 4, 42)


if __name__ == '__main__':
    unittest.main()
