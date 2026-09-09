import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "hardware_bundle.py"
SPEC = importlib.util.spec_from_file_location("hardware_bundle_direct", SCRIPT)
bundle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bundle)


class HardwareBundleTest(unittest.TestCase):
    def test_missing_comparator_case_remains_explicitly_unpaired(self):
        rows = [
            {
                "case_id": "v1:only-built-in",
                "benchmark": "only-built-in",
                "parameters": {},
                "arm": "built-in",
                "pass": 1,
                "score": 2.0,
                "score_error": 0.1,
                "unit": "us/op",
                "raw_samples": [[2.0]],
            },
            {
                "case_id": "v1:different-comparator-case",
                "benchmark": "different-comparator-case",
                "parameters": {},
                "arm": "openblas",
                "pass": 1,
                "score": 1.0,
                "score_error": 0.1,
                "unit": "us/op",
                "raw_samples": [[1.0]],
            },
        ]

        aggregates = bundle.aggregate_rows(rows)

        comparator = next(row for row in aggregates if row["arm"] == "openblas")
        self.assertIsNone(comparator["ratio_koblas_to_comparator"])
        self.assertIsNone(comparator["ratio_uncertainty"])

    def test_incomplete_bundle_fails_without_preflight_or_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            with self.assertRaisesRegex(bundle.ReportError, "bundle is missing"):
                bundle.validate_directory(directory)

    def test_archive_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "unsafe.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                payload = b"outside"
                member = tarfile.TarInfo("../outside.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

            with self.assertRaisesRegex(bundle.ReportError, "unsafe archive path"):
                with bundle.bundle_directory(archive_path):
                    pass

    def test_raw_arm_substitution_is_rejected_by_pure_validation(self):
        entry = {
            "benchmark": "example",
            "mode": "avgt",
            "params": {"sparseArm": "built-in"},
            "warmupIterations": 1,
            "measurementIterations": 1,
            "warmupTime": "20 ms",
            "measurementTime": "20 ms",
            "forks": 1,
            "threads": 1,
            "primaryMetric": {"score": 1.0, "scoreUnit": "us/op"},
        }

        with self.assertRaisesRegex(bundle.ReportError, "declared onemkl arm"):
            bundle.validate_raw_entry(
                entry,
                "onemkl",
                {"warmups": 1, "iterations": 1, "iteration_time_ms": 20, "forks": 1, "threads": 1},
                "smoke",
            )


if __name__ == "__main__":
    unittest.main()
