import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "hardware_report.py"
SPEC = importlib.util.spec_from_file_location("hardware_report", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(report)


def jmh_row(arm="built-in", score=2.0):
    return {
        "benchmark": "com.eignex.koblas.bench.Level3Benchmark.gemm",
        "mode": "avgt",
        "threads": 1,
        "forks": 1,
        "warmupIterations": 1,
        "warmupTime": "200 ms",
        "measurementIterations": 1,
        "measurementTime": "200 ms",
        "params": {"denseArm": arm, "n": "64"},
        "primaryMetric": {"score": score, "scoreError": 0.2, "scoreUnit": "us/op", "rawData": [[score]]},
    }


def write_bundle(root, *, arm="built-in", log="ok", pass_two_rows=None):
    raw = root / "raw" / arm
    raw.mkdir(parents=True)
    rows = [jmh_row(arm)]
    for number, pass_rows in ((1, rows), (2, pass_two_rows or rows)):
        (raw / f"pass-{number}.json").write_text(json.dumps(pass_rows))
        (raw / f"pass-{number}.log").write_text(log)
    pass_rows = report.load_pass(raw / "pass-1.json", arm, 1, 1)
    pass_rows.extend(report.load_pass(raw / "pass-1.json", arm, 2, 1))
    aggregates = report.aggregate_rows(pass_rows)
    (root / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "mode": "standard",
        "arms": [arm],
        "expected_cases": {arm: 1},
        "workload": {
            "version": 1,
            "settings": {"warmups": 1, "iterations": 1, "iteration_time_ms": 200, "forks": 1, "threads": 1},
        },
    }))
    report.write_rows(root, aggregates)
    (root / "summary.txt").write_text("summary\n")
    report.checksums(root)


def write_matched_bundle(root):
    all_pass_rows = []
    for arm, score in (("built-in", 2.0), ("openblas", 1.0)):
        raw = root / "raw" / arm
        raw.mkdir(parents=True)
        for number in (1, 2):
            path = raw / f"pass-{number}.json"
            path.write_text(json.dumps([jmh_row(arm, score)]))
            (raw / f"pass-{number}.log").write_text("ok")
            all_pass_rows.extend(report.load_pass(path, arm, number, 1))
    (root / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "mode": "standard",
        "arms": ["built-in", "openblas"],
        "expected_cases": {"built-in": 1, "openblas": 1},
        "workload": {
            "version": 1,
            "settings": {"warmups": 1, "iterations": 1, "iteration_time_ms": 200, "forks": 1, "threads": 1},
        },
    }))
    report.write_rows(root, report.aggregate_rows(all_pass_rows))
    (root / "summary.txt").write_text("summary\n")
    report.checksums(root)


class HardwareReportTest(unittest.TestCase):
    def test_current_onemkl_runtime_soname_is_probed(self):
        self.assertIn("libmkl_rt.so.3", report.COMPARATOR_LIBRARY_CANDIDATES["onemkl"])

    def test_case_id_excludes_arm(self):
        built_in = report.stable_case_id(jmh_row(), 1)
        openblas = report.stable_case_id(jmh_row("openblas"), 1)

        self.assertEqual(built_in, openblas)
        self.assertEqual(built_in, "v1:com.eignex.koblas.bench.Level3Benchmark.gemm[n=64]")

    def test_v1_case_id_normalizes_only_the_historical_triangle_default(self):
        row = jmh_row()
        row["benchmark"] = f"{report.V1_SPARSE_PRODUCT_BENCHMARK}.preparedGemv"
        row["params"]["triangleVariant"] = report.V1_TRIANGLE_VARIANT

        self.assertEqual(
            report.stable_case_id(row, 1),
            f"v1:{report.V1_SPARSE_PRODUCT_BENCHMARK}.preparedGemv[n=64]",
        )

        row["params"]["triangleVariant"] = "lower-trans-nonunit"
        with self.assertRaisesRegex(report.ReportError, "developer triangular variant"):
            report.stable_case_id(row, 1)

    def test_retained_onemkl_report_preserves_historical_v1_case_ids(self):
        results = SCRIPT.parents[1] / "results"
        historical = results / "hardware-standard-18286e42-jvm-20260909.tar.gz"
        one_mkl = results / "onemkl-sparse-parity-b1b0b167-jvm-20260909.tar.gz"

        def built_in_ids(archive):
            with report.bundle_directory(archive) as bundle:
                metadata = json.loads((bundle / "metadata.json").read_text())
                raw = report.raw_entries(bundle / "raw" / "built-in" / "pass-1.json")
                return {
                    report.stable_case_id(row, metadata["workload"]["version"])
                    for row in raw
                }

        expected = built_in_ids(historical)
        self.assertEqual(len(expected), 83)
        self.assertEqual(built_in_ids(one_mkl), expected)

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

    def test_validator_recomputes_aggregate_rows_and_ratios(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_matched_bundle(bundle)
            document = json.loads((bundle / "rows.json").read_text())
            openblas = next(row for row in document["rows"] if row["arm"] == "openblas")
            openblas["score"] = 123456789.0
            openblas["pass_scores"] = [123456789.0, 123456789.0]
            openblas["ratio_koblas_to_comparator"] = 0.000001
            (bundle / "rows.json").write_text(json.dumps(document))
            report.checksums(bundle)

            with self.assertRaisesRegex(report.ReportError, "does not match raw measurements"):
                report.validate_directory(bundle)

    def test_validator_rejects_raw_arm_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle, arm="openblas")
            raw = bundle / "raw" / "openblas" / "pass-1.json"
            document = json.loads(raw.read_text())
            document[0]["params"]["denseArm"] = "built-in"
            raw.write_text(json.dumps(document))
            report.checksums(bundle)

            with self.assertRaisesRegex(report.ReportError, "declared openblas arm"):
                report.validate_directory(bundle)

    def test_validator_rejects_raw_measurement_setting_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            write_bundle(bundle)
            raw = bundle / "raw" / "built-in" / "pass-2.json"
            document = json.loads(raw.read_text())
            document[0]["measurementTime"] = "500 ms"
            raw.write_text(json.dumps(document))
            report.checksums(bundle)

            with self.assertRaisesRegex(report.ReportError, "measurementTime"):
                report.validate_directory(bundle)

    @mock.patch.object(report.subprocess, "run")
    def test_native_metadata_uses_gradle_resolved_compiler(self, run):
        run.side_effect = [
            report.subprocess.CompletedProcess(
                [], 0,
                "kind=Kotlin/Native\nexecutable=/custom/kotlin/bin/konanc\ndistribution=custom-kotlin\nruntime=native executable\n",
                "",
            ),
            report.subprocess.CompletedProcess([], 0, "info: kotlinc-native 2.4.10\n", ""),
        ]

        metadata = report.native_metadata()

        self.assertEqual(metadata["distribution"], "custom-kotlin")
        self.assertEqual(metadata["compiler"], "info: kotlinc-native 2.4.10")
        self.assertNotIn("executable", metadata)
        self.assertEqual(run.call_args_list[1].args[0], ["/custom/kotlin/bin/konanc", "-version"])


if __name__ == "__main__":
    unittest.main()
