"""Behavior checks for paired capture fitting, independent of installed BLAS libraries."""
import importlib.util
import io
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
        self.assertEqual(len(holdout), len(set(holdout)))
        self.assertTrue({"dot+3+uniform", "dot+3+uniform+stride=2", "axpy+3+uniform",
                         "sum+3+uniform"}.issubset(holdout))
        native = tune.native_tail_cases()
        self.assertEqual(len(native), len(set(native)))
        self.assertTrue(set(native).isdisjoint(training + holdout))
        self.assertTrue(all(int(case.split('+')[1]) * 8 > 32 * 1024 ** 2 for case in native))
        native_holdout = tune.holdout_cases("native")
        self.assertTrue(set(native_holdout).isdisjoint(training + native))
        for stride in ("", "+stride=2", "+stride=-1", "+stride=-2"):
            lengths = lambda cases: [int(case.split('+')[1]) for case in cases
                                     if case.startswith("dot+") and case.endswith("uniform" + stride)]
            self.assertGreater(max(lengths(native_holdout)), max(lengths(training + native)))

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
        shapes = {"dot": ("Dot", 32, 1, 1, False, 1, True), "asum": ("Asum", 32, 1, 1, False, 1, True),
                  "gemm": ("Gemm", 32, 32, 32, False, 3, True)}
        def capture(costs):
            paired = {case: [cost] * 3 for case, cost in costs.items()}
            return SimpleNamespace(paired=paired, shapes=shapes, by_cpu={case: {"CPU A": values} for case, values in paired.items()}, cache_by_cpu={})
        rules, precedence = tune.compose({"onemkl": capture({"dot": 0.8, "asum": 0.8, "gemm": 0.5}),
                                          "openblas": capture({"dot": 0.5, "asum": 0.5, "gemm": 0.8}),
                                          "jvm-simd": capture({"dot": 0.6})}, [("aocl", "openblas")])

        self.assertEqual(["openblas", "onemkl"], precedence["Dot"]["sources"])
        self.assertEqual(["onemkl", "openblas"], precedence["Gemm"]["sources"])
        self.assertEqual(["openblas", "onemkl", "aocl"], [rule[0] for rule in rules if rule[1] == "Asum,Dot"])
        for operation, source in {"Asum": "openblas", "Dot": "openblas", "Gemm": "onemkl"}.items():
            self.assertEqual(source, next(rule[0] for rule in rules if operation in rule[1].split(',')))
        self.assertEqual(["jvm-simd", "Dot"], rules[-2][:2])
        self.assertEqual(["jvm-simd", "Panels,Products,Triangular"], rules[-1][:2])
        self.assertEqual([rule[1:] for rule in rules if rule[0] == "openblas"],
                         [rule[1:] for rule in rules if rule[0] == "aocl"])
        self.assertGreater(next(i for i, rule in enumerate(rules) if rule[0] == "aocl"),
                           max(i for i, rule in enumerate(rules) if rule[0] in ("onemkl", "openblas")))

    def test_emitted_fallbacks_are_estimates_with_exact_retained_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            output = root / "fitted"
            arguments = ["tune-engine.py", "--runtime", "native", "--backend", "onemkl", "--results", str(root),
                         "--fallback", "openblas=onemkl", "--output", str(output)]

            with patch.object(sys, "argv", arguments):
                tune.main()

            evidence = json.loads((output / "evidence.json").read_text())
            self.assertEqual({"openblas": "onemkl"}, evidence["best_effort_sources"])
            self.assertEqual({"tune-engine.py", "tuning_capture.py"}, set(evidence["calibration_sources"]))
            self.assertEqual({"onemkl"}, {rule[0] for rule in evidence["fitted_rules"]})
            profile = (output / "profile.tsv").read_text().splitlines()
            self.assertIn("Unmeasured best effort: openblas from onemkl", profile[0])
            self.assertEqual([[str(value) for value in rule] for rule in evidence["routing_rules"]],
                             [line.split('\t') for line in profile[1:]])
            for choice in ("typo=onemkl", "openblas=typo", "onemkl=onemkl", "openblas=jvm-simd"):
                with self.subTest(choice=choice), patch.object(sys, "argv", arguments[:-4] + ["--fallback", choice, "--output", str(output)]), \
                     patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                    tune.main()

    def test_vendor_precedence_ignores_losses_outside_the_overlapping_routed_region(self):
        shapes = {str(n): ("Dot", n, 1, 1, False, 1, True) for n in (16, 32, 64)}
        def capture(costs):
            paired = {case: [cost] * 3 for case, cost in zip(shapes, costs)}
            return SimpleNamespace(paired=paired, shapes=shapes, by_cpu={case: {"CPU": ratios} for case, ratios in paired.items()}, cache_by_cpu={})

        _, precedence = tune.compose({"openblas": capture([3.0, 3.0, 0.6]), "onemkl": capture([1.5, 1.5, 0.7])})

        self.assertEqual(["openblas", "onemkl"], precedence["Dot"]["sources"])
        self.assertEqual(["64"], precedence["Dot"]["cases"])

    def test_profile_validation_gates_hash_disjointness_and_control_regressions(self):
        for failure in (None, "hash", "overlap", "regression", "rules", "missing", "catalog"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rounds(root)
                case = "dot+64+uniform+stride=1+workspace=retained"
                validation_cases = root / "holdouts.txt"
                validation_cases.write_text(f"# expected workloads\n  {case}  \n" + ("sum+3+uniform\n" if failure == "missing" else ""))
                profile = root / "profile.tsv"
                rule = ["onemkl", "Dot", 0, tune.MAX, 0, tune.MAX, 0, tune.MAX, 0, "*", "*"]
                rules = [rule, ["openblas", *rule[1:]]]
                profile.write_text("koblas-profile-v3\ttest\tnative\tLinux\tX86_64\t*\tAvx2\t*\tfixture\ttest\n" +
                                   "\n".join("\t".join(map(str, row)) for row in rules) + "\n")
                if failure == "rules":
                    profile.write_text(profile.read_text().replace("onemkl\tDot\t0", "onemkl\tDot\t1"))
                digest = hashlib.sha256(profile.read_bytes()).hexdigest()
                (root / "evidence.json").write_text(json.dumps(dict(routing_rules=rules, fitted_rules=[rule],
                    best_effort_sources={"openblas": "onemkl"}, sources=dict(onemkl=dict(
                    paired_ratios={case if failure == "overlap" else "dot+32+uniform": [0.5] * 3})))))
                for path in root.glob("pass-*"):
                    metadata = (path / "metadata.txt").read_text().replace("[onemkl]", "[native-default]")
                    (path / "metadata.txt").write_text(metadata.replace("dirty=false", f"engine_profile_sha256={digest}\ndirty=false"))
                    for name in ("native", "onemkl"):
                        csv = path / f"{name}.csv"
                        text = csv.read_text().replace("dot+32+uniform", "dot+64+uniform+workspace=retained+stride=1").replace("Dot:32:", "Dot:64:")
                        if name == "onemkl":
                            if failure == "regression":
                                previous = 5 * 10 ** (int(path.name.removeprefix("pass-")) - 1)
                                text = text.replace(f",{previous},onemkl", f",{previous * (0.5 if path.name == 'pass-1' else 3)},onemkl")
                            csv = path / "native-default.csv"
                        csv.write_text(text)
                if failure == "hash":
                    profile.write_text(profile.read_text() + "# changed profile\n")
                if failure:
                    with self.assertRaises(ValueError):
                        tune.validate_profile([root], profile, root, "native",
                                              validation_cases=None if failure == "catalog" else validation_cases)
                else:
                    with patch.object(sys, "argv", ["tune-engine.py", "--validate-results", str(root), "--profile", str(profile),
                                                    "--output", str(root / "validated"), "--validation-cases", str(validation_cases)]):
                        tune.main()
                    self.assertEqual({}, json.loads((root / "validated/validation.json").read_text())["regressions"])


if __name__ == "__main__":
    unittest.main()
