import csv
from pathlib import Path
import tempfile
import unittest

from summarize_report import summarize_benchmark, summarize_cpu, summarize_report


HEADERS = [
    ['run', 'id', 'implementation', 'pass', 'unit', 'source_commit', 'dirty', 'runtime', 'threads',
     'warmups', 'target_ns', 'harness', 'warmup_target_ns', 'forks'],
    ['case', 'id', 'run_id', 'case', 'status', 'comparison_kind', 'timing_mode', 'actual_kernel'],
    ['sample', 'case_id', 'fork', 'sample', 'operations', 'elapsed_ns', 'ns_per_op'],
]
RECORDS = [
    ['run', '1', 'jvm-simd', '1', 'ns', 'abc', 'false', 'runtime, with comma', '1', '5', '200000000', 'jmh', '200000000', '2'],
    ['case', '1', '1', 'dot+4+uniform', 'ok', 'direct', 'arithmetic', 'simd'],
    ['sample', '1', '1', '1', '10', '10', '1'],
    ['sample', '1', '1', '2', '10', '90', '9'],
    ['sample', '1', '2', '3', '10', '30', '3'],
    ['sample', '1', '2', '4', '10', '50', '5'],
    ['case', '2', '1', 'spdot+4+sparse-uniform', 'unsupported', 'unsupported', 'arithmetic', 'unavailable'],
]
METADATA = '''status=complete
started_at=2026-09-12T00:00:00Z
jvm-scalar_started_at=2026-09-12T00:00:03Z
jvm-c_started_at=2026-09-12T00:00:04Z
jvm-simd_started_at=2026-09-12T00:00:05Z
native_started_at=2026-09-12T00:00:06Z
vendors_started_at=2026-09-12T00:00:07Z
completed_at=2026-09-12T00:00:08Z
'''


class SummarizeReportTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / 'raw.csv'
        self.destination = self.root / 'summary.csv'

    def write_benchmark(self, records=RECORDS):
        with self.source.open('w', newline='') as stream:
            csv.writer(stream).writerows(HEADERS + records)

    def test_one_summary_per_case_preserves_provenance_and_distribution(self):
        self.write_benchmark()
        original = self.source.read_bytes()
        summarize_benchmark(self.source, self.destination)
        with self.destination.open(newline='') as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[2], RECORDS[0])
        self.assertEqual(rows[3][-5:], ['4', '2', '4', '1', '9'])
        self.assertEqual(rows[4][-5:], ['0', '0', '', '', ''])
        self.assertEqual(self.source.read_bytes(), original)

    def test_invalid_samples_are_rejected(self):
        for value in ('nan', 'inf', '0', '-1'):
            with self.subTest(value=value):
                records = [row.copy() for row in RECORDS]
                records[2][-1] = value
                self.write_benchmark(records)
                with self.assertRaises(ValueError):
                    summarize_benchmark(self.source, self.destination)

    def test_duplicate_samples_are_rejected(self):
        self.write_benchmark(RECORDS + [RECORDS[2]])
        with self.assertRaises(ValueError):
            summarize_benchmark(self.source, self.destination)

    def test_missing_fork_is_rejected(self):
        self.write_benchmark([row for row in RECORDS if not (row[0] == 'sample' and row[2] == '2')])
        with self.assertRaises(ValueError):
            summarize_benchmark(self.source, self.destination)

    def test_cpu_phases_preserve_missing_readings(self):
        self.source.write_text('elapsed_s,cpu_percent\n1,10\n2,\n3,20\n4,\n5,30\n6,40\n7,50\n8,99\n')
        summarize_cpu(self.source, METADATA, self.destination)
        with self.destination.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]['readings'], '2')
        self.assertEqual(rows[0]['available_readings'], '1')
        self.assertEqual(rows[0]['mean_cpu_percent'], '10')
        self.assertEqual(rows[1]['mean_cpu_percent'], '20')
        self.assertEqual(rows[2]['mean_cpu_percent'], '')
        self.assertEqual(rows[-1]['mean_cpu_percent'], '50')

    def test_export_keeps_raw_capture_and_refuses_overwrite(self):
        self.source = self.root / 'jvm-simd.csv'
        self.write_benchmark()
        (self.root / 'metadata.txt').write_text(METADATA)
        (self.root / 'cpu.csv').write_text('elapsed_s,cpu_percent\n1,10\n')
        output = self.root / 'report'
        summarize_report(self.root, output)
        self.assertEqual({p.name for p in output.iterdir()}, {'metadata.txt', 'cpu-summary.csv', 'jvm-simd-summary.csv'})
        self.assertEqual((output / 'metadata.txt').read_text(), METADATA)
        self.assertTrue(self.source.exists())
        with self.assertRaises(FileExistsError):
            summarize_report(self.root, output)

    def test_incomplete_capture_is_rejected(self):
        (self.root / 'metadata.txt').write_text(METADATA.replace('status=complete', 'status=incomplete'))
        with self.assertRaises(ValueError):
            summarize_report(self.root, self.root / 'report')
        self.assertFalse((self.root / 'report').exists())


if __name__ == '__main__':
    unittest.main()
