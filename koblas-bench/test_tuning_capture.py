"""Pairing, target identity and failed-attempt retention for calibration inputs."""
import json
import tempfile
import unittest
from pathlib import Path
from tuning_capture import measurements


def rounds(root, model="measured CPU", runtime="native"):
    for index, (base, candidate) in enumerate(((10, 5), (100, 50), (1000, 500)), 1):
        path = root / f"pass-{index}"
        path.mkdir(parents=True)
        metadata = ("status=complete\n[capture]\nrequested_samples=5\ntarget_ns=200000000\n"
                    "warmups=20\nrequested_jvm_forks=2\ndirty=false\nsource_commit=" + "a" * 40 + "\n")
        metadata += f"[hardware]\nmodel name={model}\n"
        for name, timing in (("native", base), ("onemkl", candidate)):
            metadata += (f"[{name}]\nprofile_runtime={runtime}\nprofile_os=Linux\nprofile_arch=X86_64\n"
                         f"profile_cpu_model={model}\nimplementation={name}/caps[{'portable' if name == 'native' else name}:Dot]\n")
            (path / f"{name}.csv").write_text(
                "case,status,timing_mode,routing_key,samples,median_ns,provider\n"
                f"dot+32+uniform,ok,arithmetic,Dot:32:1:1:false:1:true,5,{timing},{name}\n")
        (path / "metadata.txt").write_text(metadata)


def lab_row(index, arm, problem="dot", status="DONE"):
    return dict(index=index, arm=arm, problem=dict(problem=problem), seed=None, repeat=0, status=status,
                record=dict(status="success", artifacts=[dict(path=f"report/pass-{i}/{name}")
                    for i in (1, 2, 3) for name in ("metadata.txt", "native.csv", "onemkl.csv")]))


def lab_index(root, rows):
    (root / "lab-cases.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (root / "lab-job.json").write_text('{"id":42,"status":"DONE"}')
    (root / "lab-execution.json").write_text('[{"failedAttempt":{"cpuMillis":1234}}]')


class TuningCaptureTest(unittest.TestCase):
    def test_pairs_round_ratios_and_uses_runtime_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            capture = measurements(root, "native", "onemkl")
            self.assertEqual([0.5] * 3, capture.paired["dot+32+uniform"])
            self.assertEqual(9, len(capture.hashes))
            self.assertEqual(("Dot", 32, 1, 1, False, 1, True), capture.shapes["dot+32+uniform"])
            self.assertEqual(("native", "Linux", "X86_64", "measured CPU"), capture.target("native"))
            for target in (("jvm",), ("native", "MacOs"), ("native", "Linux", "Arm64"),
                           ("native", "Linux", "X86_64", "another CPU")):
                with self.assertRaises(ValueError):
                    capture.target(*target)

    def test_unknown_runtime_cpu_needs_explicit_wildcard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            for path in root.glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("profile_cpu_model=measured CPU", "profile_cpu_model=*"))
            capture = measurements(root, "native", "onemkl")
            with self.assertRaises(ValueError):
                capture.target("native")
            self.assertEqual("*", capture.target("native", cpu="*")[-1])

    def test_rejects_missing_runtime_identity_and_smoke(self):
        for previous, replacement in (("profile_cpu_model=measured CPU", "cpu_model=hardware only"),
                                      ("target_ns=200000000", "target_ns=1000000"),
                                      ("requested_samples=5", "requested_samples=1")):
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rounds(root)
                path = root / "pass-1/metadata.txt"
                path.write_text(path.read_text().replace(previous, replacement))
                with self.assertRaises(ValueError):
                    measurements(root, "native", "onemkl")

    def test_validation_rejects_empty_untimed_or_inconsistent_cases(self):
        for replacement in ("", "dot+32+uniform,unsupported,arithmetic,Dot:32:1:1:false:1:true,,onemkl\n",
                            "dot+64+uniform,ok,arithmetic,Dot:64:1:1:false:1:true,5,onemkl\n"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rounds(root)
                for path in root.glob("*/onemkl.csv"):
                    path.write_text(path.read_text().splitlines()[0] + "\n" + replacement)
                with self.assertRaises(ValueError):
                    measurements(root, "native", "onemkl", strict=True)

    def test_accepts_only_complete_rows_and_retains_failure_cost_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [lab_row(0, "base"), lab_row(1, "candidate"),
                    lab_row(2, "base", "broken"), lab_row(3, "candidate", "broken", "FAILED")]
            lab_index(root, rows)
            rounds(root / "cases/1/out/report")
            # Tempting timings in an excluded row must never enter a fitted crossover.
            rounds(root / "cases/3/out/report", "another CPU")
            capture = measurements(root, "native", "onemkl", arm="candidate")
            self.assertEqual([1], capture.lab["accepted_rows"])
            self.assertEqual({"measured CPU"}, capture.models)
            self.assertEqual(4, len(capture.lab["rows"]))
            self.assertEqual(1234, capture.lab["resources"]["lab-execution.json"][0]["failedAttempt"]["cpuMillis"])
            with self.assertRaises(ValueError):
                measurements(root, "native", "onemkl")

    def test_missing_arm_and_escaping_artifacts_cannot_supply_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [lab_row(0, "base"), lab_row(1, "candidate", "another problem")]
            lab_index(root, rows)
            with self.assertRaises(ValueError):
                measurements(root, "native", "onemkl", arm="candidate")
            rows[1]["problem"]["problem"] = "dot"
            rows[1]["record"]["artifacts"][0]["path"] = "../../../../outside/pass-1/metadata.txt"
            lab_index(root, rows)
            with self.assertRaises(ValueError):
                measurements(root, "native", "onemkl", arm="candidate")

    def test_different_cpu_models_stay_in_separate_strata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [lab_row(0, "candidate"), lab_row(1, "candidate", "dot repetition")]
            lab_index(root, rows)
            rounds(root / "cases/0/out/report", "CPU A")
            rounds(root / "cases/1/out/report", "CPU B")
            capture = measurements(root, "native", "onemkl")
            self.assertEqual({"CPU A": [0.5] * 3, "CPU B": [0.5] * 3}, capture.by_cpu["dot+32+uniform"])
            with self.assertRaises(ValueError):
                capture.target("native")
            self.assertEqual("*", capture.target("native", cpu="*")[-1])

    def test_vendor_competes_with_the_faster_owned_arm_in_each_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root, runtime="jvm")
            for path in root.glob("pass-*"):
                portable = (path / "native.csv").read_text()
                header, row = portable.splitlines()
                values = row.split(',')
                values[-2] = {"pass-1": "100", "pass-2": "25", "pass-3": "1000"}[path.name]
                simd = header + "\n" + ','.join(values) + "\n"
                (path / "jvm-scalar.csv").write_text(portable)
                (path / "jvm-default.csv").write_text(simd)
                metadata = path / "metadata.txt"
                text = metadata.read_text()
                identity = text[text.index("[native]"):text.index("[onemkl]")]
                metadata.write_text(text + identity.replace("[native]", "[jvm-scalar]") +
                                    identity.replace("[native]", "[jvm-default]"))
            capture = measurements(root, "jvm-owned", "onemkl", owned_target="jvm-default")
            self.assertEqual([0.5, 2.0, 0.5], capture.paired["dot+32+uniform"])

    def test_capture_rounds_cannot_mix_source_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            path = root / "pass-3/metadata.txt"
            path.write_text(path.read_text().replace("a" * 40, "b" * 40))
            with self.assertRaises(ValueError):
                measurements(root, "native", "onemkl")


if __name__ == "__main__":
    unittest.main()
