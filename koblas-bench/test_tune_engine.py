"""Behavior checks for paired capture fitting, independent of installed BLAS libraries."""
import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from test_tuning_capture import rounds

spec = importlib.util.spec_from_file_location("tune", Path(__file__).with_name("tune-engine.py"))
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)


class TuneEngineTest(unittest.TestCase):
    def test_generated_grid_is_unique_and_disjoint_from_holdouts(self):
        training = tune.training_cases()
        holdout = tune.holdout_cases()
        self.assertEqual(len(training), len(set(training)))
        self.assertTrue(set(training).isdisjoint(holdout))
        self.assertTrue(set(tune.supplementary_cases()).issubset(training))
        self.assertTrue(set(tune.tail_cases()).issubset(training))
        self.assertTrue(set(tune.anchor_cases()).issubset(training))
        self.assertEqual(len(holdout), len(set(holdout)))

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

    def test_one_cpu_model_cannot_hide_a_nonwinning_model(self):
        paired = {"dot": [0.5] * 30 + [0.99] * 3}
        rules = tune.fit(paired, "onemkl", {"dot": ("Dot", 128, 1, 1, False, 1, True)},
                         {"dot": {"CPU A": [0.5] * 30, "CPU B": [0.99] * 3}})
        self.assertEqual([], rules)

    def test_open_ended_vector_rule_requires_two_winning_beyond_cache_lengths(self):
        shapes = {str(n): ("Dot", n, 1, 1, False, 1, True) for n in (32, 128, 256)}
        paired = {case: [0.5] * 3 for case in shapes}
        self.assertEqual(tune.MAX, tune.fit(paired, "onemkl", shapes, cache_bytes=512)[0][3])
        self.assertEqual(256, tune.fit(paired, "onemkl", shapes, cache_bytes=1024)[0][3])
        self.assertEqual(256, tune.fit(paired, "onemkl", shapes)[0][3])
        self.assertEqual(tune.MAX, tune.fit(paired, "jvm-simd", shapes)[0][3])
        paired["128"] = [1.5] * 3
        self.assertEqual(256, tune.fit(paired, "jvm-simd", shapes)[0][3])

    def test_matrix_work_and_extents_exclude_skinny_and_small_losses(self):
        shapes = {"small": ("Gemm", 8, 8, 8, False, 3, True),
                  "skinny": ("Gemm", 4, 512, 512, False, 3, True),
                  "medium": ("Gemm", 32, 32, 32, False, 3, True),
                  "large": ("Gemm", 64, 64, 64, False, 3, True)}
        paired = {case: [0.5 if case in ("medium", "large") else 1.5] * 3 for case in shapes}
        rule, = tune.fit(paired, "onemkl", shapes)
        self.assertEqual([32, tune.MAX, 32, tune.MAX, 32, tune.MAX, 32 ** 3], rule[2:9])

    def test_deep_depth_bands_stop_at_observed_losses_and_split_borrowed_rules(self):
        shapes = {str(depth): ("Gemm", 16, 16, depth, False, 3, True)
                  for depth in (512, 1024, 2048, 4096, 8192)}
        shapes["borrowed"] = ("Gemm", 16, 16, 1024, True, 3, True)
        paired = {case: [0.5 if case in ("1024", "2048", "borrowed") else 1.5] * 3 for case in shapes}
        rules = tune.fit(paired, "onemkl", shapes)
        self.assertEqual([[1024, 2048], [1024, 1024]], [rule[6:8] for rule in rules])
        self.assertEqual(["false", "true"], [rule[10] for rule in rules])

    def test_complete_profiles_use_measured_vendor_precedence_and_explicit_owned_rules(self):
        shapes = {"dot": ("Dot", 32, 1, 1, False, 1, True), "gemm": ("Gemm", 32, 32, 32, False, 3, True)}
        def capture(costs):
            paired = {case: [cost] * 3 for case, cost in costs.items()}
            return SimpleNamespace(paired=paired, shapes=shapes, by_cpu={case: {"CPU A": values} for case, values in paired.items()}, cache_by_cpu={})
        rules, precedence = tune.compose({"onemkl": capture({"dot": 0.8, "gemm": 0.5}),
                                          "openblas": capture({"dot": 0.5, "gemm": 0.8}),
                                          "jvm-simd": capture({"dot": 0.6})})

        self.assertEqual(["openblas", "onemkl"], precedence["Dot"]["sources"])
        self.assertEqual(["onemkl", "openblas"], precedence["Gemm"]["sources"])
        self.assertEqual(["jvm-simd", "Dot"], rules[-2][:2])
        self.assertEqual(["jvm-simd", "Panels,Products,Triangular"], rules[-1][:2])

    def test_vendor_precedence_ignores_losses_outside_the_overlapping_routed_region(self):
        shapes = {str(n): ("Dot", n, 1, 1, False, 1, True) for n in (16, 32, 64)}
        def capture(costs):
            paired = {case: [cost] * 3 for case, cost in zip(shapes, costs)}
            return SimpleNamespace(paired=paired, shapes=shapes, by_cpu={case: {"CPU": ratios} for case, ratios in paired.items()}, cache_by_cpu={})

        _, precedence = tune.compose({"openblas": capture([3.0, 3.0, 0.6]), "onemkl": capture([1.5, 1.5, 0.7])})

        self.assertEqual(["openblas", "onemkl"], precedence["Dot"]["sources"])
        self.assertEqual(["64"], precedence["Dot"]["cases"])

    def test_profile_validation_gates_hash_disjointness_and_control_regressions(self):
        for failure in (None, "hash", "overlap", "regression"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rounds(root)
                profile = root / "profile.tsv"
                profile.write_text("koblas-profile-v2\ttest\tnative\tLinux\tX86_64\t*\tAvx2\tfixture\ttest\n"
                                   "onemkl\tDot\t0\t2147483647\t0\t2147483647\t0\t2147483647\t0\t*\t*\n")
                digest = hashlib.sha256(profile.read_bytes()).hexdigest()
                (root / "evidence.json").write_text(json.dumps(dict(sources=dict(onemkl=dict(
                    paired_ratios={"dot+64+uniform" if failure == "overlap" else "dot+32+uniform": [0.5] * 3})))))
                for path in root.glob("pass-*"):
                    metadata = (path / "metadata.txt").read_text().replace("[onemkl]", "[native-default]")
                    (path / "metadata.txt").write_text(metadata.replace("dirty=false", f"engine_profile_sha256={digest}\ndirty=false"))
                    for name in ("native", "onemkl"):
                        csv = path / f"{name}.csv"
                        text = csv.read_text().replace("dot+32", "dot+64").replace("Dot:32:", "Dot:64:")
                        if name == "onemkl":
                            if failure == "regression":
                                previous = 5 * 10 ** (int(path.name.removeprefix("pass-")) - 1)
                                text = text.replace(f",{previous},onemkl", f",{previous * 3},onemkl")
                            csv = path / "native-default.csv"
                        csv.write_text(text)
                if failure == "hash":
                    profile.write_text(profile.read_text() + "# changed profile\n")
                if failure:
                    with self.assertRaises(ValueError):
                        tune.validate_profile([root], profile, root, "native")
                else:
                    with patch.object(sys, "argv", ["tune-engine.py", "--validate-results", str(root), "--profile", str(profile),
                                                    "--output", str(root / "validated")]):
                        tune.main()
                    self.assertEqual({}, json.loads((root / "validated/validation.json").read_text())["regressions"])


if __name__ == "__main__":
    unittest.main()
