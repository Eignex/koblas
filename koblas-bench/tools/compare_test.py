import csv
import importlib.util
import subprocess
import tempfile
import unittest
import sys
sys.dont_write_bytecode = True
from pathlib import Path

TOOLS = Path(__file__).parent
spec = importlib.util.spec_from_file_location('compare', TOOLS / 'compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.row = dict.fromkeys(compare.REQUIRED, '1')
        self.row.update(schema='4', case='old-kernel+shape', logical_id='gemm-add-v1+shape',
                        configuration='tile-four', physical_work='tiles=4', timing_mode='prepacked-compute',
                        ns_per_op='2', status='ok', comparison_kind='direct')

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, rows):
        path = self.directory / name
        with path.open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=sorted(compare.REQUIRED))
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    def run_compare(self, left, right, mode, timing=None):
        return subprocess.run(['python3', str(TOOLS / 'compare.py'), '--require-compatible', '--mode', mode,
                               self.write('base.csv', left), self.write('candidate.csv', right)] +
                              (['--timing', timing] if timing else []), capture_output=True, text=True)

    def test_logical_mode_keeps_layout_pairs_separate(self):
        other = self.row | dict(configuration='tile-eight', physical_work='tiles=2', case='new-kernel+shape')
        result = self.run_compare([self.row], [self.row, other], 'logical')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(3, len(result.stdout.splitlines()))
        self.assertNotEqual(0, self.run_compare([self.row], [other], 'fixed').returncode)

    def test_fixed_identity_survives_kernel_rename(self):
        result = self.run_compare([self.row], [self.row | dict(case='new-name', actual_kernel='new-symbol')], 'fixed')
        self.assertEqual(0, result.returncode, result.stderr)

    def test_fixture_timing_and_physical_mismatches_fail(self):
        for key in ('fixture_version', 'workload_version', 'timing_mode', 'physical_work', 'threads', 'warmups', 'target_ns'):
            with self.subTest(key=key):
                self.assertNotEqual(0, self.run_compare([self.row], [self.row | {key: 'different'}], 'fixed').returncode)
        for timing in ('raw-tile', 'packing-only', 'layout-only'):
            left = self.row | dict(timing_mode=timing)
            self.assertNotEqual(0, self.run_compare([left], [left | dict(physical_work='tiles=2')], 'logical').returncode)

    def test_boundary_selector_excludes_deliberately_incompatible_raw_rows(self):
        raw = self.row | dict(timing_mode='raw-tile')
        vendor_raw = self.row | dict(timing_mode='vendor-arithmetic')
        self.assertNotEqual(0, self.run_compare([self.row, raw], [self.row, vendor_raw], 'logical').returncode)
        result = self.run_compare([self.row, raw], [self.row, vendor_raw], 'logical', 'prepacked-compute')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(2, len(result.stdout.splitlines()))

    def test_fixed_mode_excludes_policy_experiments(self):
        policy = self.row | dict(configuration='policy-v1', physical_work='policy')
        result = self.run_compare([self.row, policy], [self.row, policy], 'fixed')
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(2, len(result.stdout.splitlines()))

    def test_disjoint_reports_and_historical_schema_fail(self):
        for patch in (dict(logical_id='different'), dict(schema='3')):
            self.assertNotEqual(0, self.run_compare([self.row], [self.row | patch], 'logical').returncode)

    def test_quoted_csv_and_sample_medians(self):
        row = self.row | dict(case='quoted "case", one')
        result = self.run_compare([row | dict(ns_per_op='1'), row | dict(ns_per_op='3')],
                                  [row | dict(ns_per_op='2'), row | dict(ns_per_op='4')], 'fixed')
        self.assertEqual(0, result.returncode, result.stderr)
        records = list(csv.DictReader(result.stdout.splitlines()))
        self.assertEqual('quoted "case", one', records[0]['base_case'])
        self.assertEqual('2.0', records[0]['base_median_ns'])
        self.assertEqual('3.0', records[0]['candidate_median_ns'])


if __name__ == '__main__':
    unittest.main()
