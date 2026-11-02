"""Behavior checks for paired capture fitting, independent of installed BLAS libraries."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("tune", Path(__file__).with_name("tune-engine.py"))
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)


class TuneEngineTest(unittest.TestCase):
    def test_noisy_or_losing_larger_lengths_delay_the_crossover(self):
        paired = {"dot+32+uniform": [0.5, 0.5, 0.5], "dot+64+uniform": [0.8, 1.2, 0.8],
                  "dot+128+uniform": [0.7, 0.8, 0.9], "dot+256+uniform": [0.8, 0.8, 0.8]}
        rule, = tune.fit(paired, "onemkl", {case: ("Dot", n, 1, 1, False, 1, True) for case, n in zip(paired, (32, 64, 128, 256))})
        self.assertEqual([128, 256], rule[2:4])

    def test_contiguous_and_strided_lengths_fit_independent_preferences(self):
        paired = {"adjacent": [0.8] * 3, "strided": [1.5] * 3}
        rules = tune.fit(paired, "onemkl", {"adjacent": ("Dot", 128, 1, 1, False, 1, True),
                                          "strided": ("Dot", 128, 1, 1, False, 1, False)})
        self.assertEqual(1, len(rules))
        self.assertEqual("true", rules[0][9])

    def test_matrix_variants_must_all_win_and_shapes_stay_bounded(self):
        paired = {"gemm+16x32x64+uniform": [0.5, 0.6, 0.7],
                  "gemm+16x32x64+uniform+transA=T": [1.1, 1.1, 1.1],
                  "gemm+64x16x32+uniform+layout=borrowed": [0.5, 0.5, 0.5]}
        rule, = tune.fit(paired, "onemkl", dict(zip(paired, [("Gemm", 16, 32, 64, False, 3, True)] * 2 + [("Gemm", 64, 16, 32, True, 3, True)])))
        self.assertEqual([64, 64, 16, 16, 32, 32], rule[2:8])
        self.assertEqual("true", rule[10])

    def test_profile_cpu_comes_from_measurements_and_rejects_mixed_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(1, 4):
                path = root / f"pass-{index}"
                path.mkdir()
                (path / "metadata.txt").write_text("model name=hardware display name\nprofile_cpu_model=measured cpu\n")
            self.assertEqual("measured cpu", tune.measured_cpu(root))
            with self.assertRaises(ValueError):
                tune.measured_cpu(root, "another model")
            (root / "pass-3/metadata.txt").write_text("profile_cpu_model=different cpu\n")
            with self.assertRaises(ValueError):
                tune.measured_cpu(root)
            (root / "pass-3/metadata.txt").write_text("model name=hardware only\n")
            with self.assertRaises(ValueError):
                tune.measured_cpu(root)

    def test_validation_rejects_empty_or_untimed_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(1, 4):
                path = root / f"pass-{index}"
                path.mkdir()
                (path / "metadata.txt").write_text("status=complete\nrequested_samples=5\ntarget_ns=200000000\n")
                for name in ("native", "native-default"):
                    (path / f"{name}.csv").write_text("case,status,timing_mode,routing_key,median_ns,provider\n")
            with self.assertRaises(ValueError):
                tune.measurements(root, "native", "native-default", strict=True)
            for path in root.glob("pass-*/*.csv"):
                path.write_text(path.read_text() + "dot+32+uniform,unsupported,arithmetic,Dot:32:1:1:false:1:true,,portable\n")
            with self.assertRaises(ValueError):
                tune.measurements(root, "native", "native-default", strict=True)

    def test_unknown_runtime_cpu_requires_an_explicit_wildcard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pass-1").mkdir()
            (root / "pass-1/metadata.txt").write_text("profile_cpu_model=*\n")
            for requested in (None, "hardware display name"):
                with self.assertRaises(ValueError):
                    tune.measured_cpu(root, requested)
            self.assertEqual("*", tune.measured_cpu(root, "*"))

    def test_pairing_uses_round_ratios_and_rejects_smoke_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (base, candidate) in enumerate([(10, 5), (100, 50), (1000, 500)], 1):
                path = root / f"pass-{index}"
                path.mkdir()
                (path / "metadata.txt").write_text("status=complete\nrequested_samples=5\ntarget_ns=100000000\n")
                for name, timing in [("native", base), ("onemkl", candidate)]:
                    (path / f"{name}.csv").write_text(
                        "case,status,timing_mode,routing_key,median_ns,provider\n" + f"dot+32+uniform,ok,arithmetic,Dot:32:1:1:false:1:true,{timing},{name}\n")
            paired, hashes, shapes, providers = tune.measurements(root, "native", "onemkl")
            self.assertEqual([0.5, 0.5, 0.5], paired["dot+32+uniform"])
            self.assertEqual(9, len(hashes))
            self.assertEqual(("Dot", 32, 1, 1, False, 1, True), shapes["dot+32+uniform"])
            (root / "pass-1/metadata.txt").write_text("status=complete\nrequested_samples=1\ntarget_ns=1000000\n")
            with self.assertRaises(ValueError):
                tune.measurements(root, "native", "onemkl")


if __name__ == "__main__":
    unittest.main()
