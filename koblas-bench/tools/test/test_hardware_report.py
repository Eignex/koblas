import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "hardware_report.py"
SPEC = importlib.util.spec_from_file_location("hardware_report", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(report)


def jmh_row(arm="built-in", score=2.0):
    return {
        "benchmark": "com.eignex.koblas.bench.Level3Benchmark.gemm",
        "params": {"denseArm": arm, "n": "64"},
        "primaryMetric": {"score": score, "scoreError": 0.2, "scoreUnit": "us/op", "rawData": [[score]]},
    }


def write_bundle(root, *, log="ok", pass_two_rows=None):
    raw = root / "raw" / "built-in"
    raw.mkdir(parents=True)
    rows = [jmh_row()]
    for number, pass_rows in ((1, rows), (2, pass_two_rows or rows)):
        (raw / f"pass-{number}.json").write_text(json.dumps(pass_rows))
        (raw / f"pass-{number}.log").write_text(log)
    pass_rows = report.load_pass(raw / "pass-1.json", "built-in", 1, 1)
    pass_rows.extend(report.load_pass(raw / "pass-1.json", "built-in", 2, 1))
    aggregates = report.aggregate_rows(pass_rows)
    (root / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "arms": ["built-in"],
        "expected_cases": {"built-in": 1},
        "workload": {"version": 1},
    }))
    report.write_rows(root, aggregates)
    (root / "summary.txt").write_text("summary\n")
    report.checksums(root)


class HardwareReportTest(unittest.TestCase):
    def test_case_id_excludes_arm(self):
        built_in = report.stable_case_id(jmh_row(), 1)
        openblas = report.stable_case_id(jmh_row("openblas"), 1)

        self.assertEqual(built_in, openblas)
        self.assertEqual(built_in, "v1:com.eignex.koblas.bench.Level3Benchmark.gemm[n=64]")

    def test_missing_case_fails(self):
        with self.assertRaisesRegex(report.ReportError, "expected 2 cases"):
            report.check_pass([{"case_id": "one"}], "built-in", 2)

    def test_fork_failure_pattern(self):
        self.assertIsNotNone(report.FAILURE_PATTERN.search("EXCEPTION: <ERROR> setup failed"))

    def test_invocation_result_search_is_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "result.json").write_text("[]")
            (second / "result.json").write_text("[]")

            self.assertEqual(report.find_result(first), first / "result.json")

    def test_paired_ratio_requires_matching_case(self):
        rows = []
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pass.json"
            path.write_text(json.dumps([jmh_row(score=2.0)]))
            rows.extend(report.load_pass(path, "built-in", 1, 1))
            path.write_text(json.dumps([jmh_row("openblas", score=1.0)]))
            rows.extend(report.load_pass(path, "openblas", 1, 1))

        aggregates = report.aggregate_rows(rows)
        comparator = next(row for row in aggregates if row["arm"] == "openblas")
        self.assertEqual(comparator["ratio_koblas_to_comparator"], 2.0)

    def test_invalid_schema_fails_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "raw").mkdir()
            (bundle / "metadata.json").write_text('{"schema_version": 99}')
            (bundle / "rows.json").write_text('{"schema_version": 1, "rows": [{}]}')
            (bundle / "rows.csv").write_text("case_id\n")
            (bundle / "summary.txt").write_text("summary\n")
            files = ["metadata.json", "rows.json", "rows.csv", "summary.txt"]
            sums = [f"{hashlib.sha256((bundle / name).read_bytes()).hexdigest()}  {name}" for name in files]
            (bundle / "SHA256SUMS").write_text("\n".join(sums) + "\n")

            with self.assertRaisesRegex(report.ReportError, "unsupported"):
                report.validate_directory(bundle)

    def test_explicit_unsupported_comparator_fails(self):
        status = {"openblas": {"availability": "unsupported"}, "onemkl": {"availability": "unavailable"}}

        with self.assertRaisesRegex(report.ReportError, "unsupported"):
            report.parse_comparators("openblas", status)

    def test_validator_detects_fork_failure_in_raw_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle, log="EXCEPTION: <ERROR> setup failed")

            with self.assertRaisesRegex(report.ReportError, "fork failure"):
                report.validate_directory(bundle)

    def test_validator_detects_raw_case_mismatch(self):
        changed = jmh_row()
        changed["params"] = {"denseArm": "built-in", "n": "257"}
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle, pass_two_rows=[changed])

            with self.assertRaisesRegex(report.ReportError, "identical case IDs"):
                report.validate_directory(bundle)

    def test_validator_requires_checksum_for_every_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle)
            (bundle / "unlisted.txt").write_text("not checksummed")

            with self.assertRaisesRegex(report.ReportError, "does not cover"):
                report.validate_directory(bundle)


if __name__ == "__main__":
    unittest.main()
