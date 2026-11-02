"""Pairing, target identity and failed-attempt retention for calibration inputs."""
import json
import tempfile
import unittest
from pathlib import Path
from tuning_capture import measurements, verify_anchors


def rounds(root, model="measured CPU", runtime="native"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "source-hashes.txt").write_text("a" * 64 + " koblas/src/commonMain/kotlin/PortableEngine.kt\n")
    for index, (base, candidate) in enumerate(((10, 5), (100, 50), (1000, 500)), 1):
        path = root / f"pass-{index}"
        path.mkdir(parents=True)
        metadata = ("status=complete\n[capture]\nrequested_samples=5\ntarget_ns=200000000\n"
                    "warmups=20\nrequested_jvm_forks=2\njvm_gc=serial\njvm_heap=512m\ncpu_policy=unbound\n"
                    "koblas_threads=1\nplatform=Linux fixture 6.12 test x86_64\ndirty=false\nsource_commit=" + "a" * 40 + "\n")
        metadata += f"[hardware]\nmodel name={model}\nflags=avx2\n"
        for name, timing in (("native", base), ("onemkl", candidate)):
            metadata += (f"[{name}]\nprofile_runtime={runtime}\nprofile_os=Linux\nprofile_arch=X86_64\n"
                         f"profile_cpu_model={model}\nprofile_cpu_class=Avx2\nruntime={runtime}-version\nharness=calibrated\nwarmup_target_ns=50000000\n"
                         f"implementation={name}{'' if name == 'native' else '/oneMKL/' + str(root) + '/libmkl.so/2026.1/threads=1-confirmed'}/caps[{'portable' if name == 'native' else name}:Dot]\n")
            (path / f"{name}.csv").write_text(
                "case,status,timing_mode,routing_key,samples,median_ns,provider\n"
                f"dot+32+uniform,ok,arithmetic,Dot:32:1:1:false:1:true,5,{timing},{name}\n")
        (path / "metadata.txt").write_text(metadata)


def lab_row(index, arm, problem="dot", status="DONE"):
    return dict(index=index, arm=arm, problem=dict(problem=problem), seed=None, repeat=0, status=status,
                record=dict(status="success", artifacts=[dict(path=f"report/pass-{i}/{name}")
                    for i in (1, 2, 3) for name in ("metadata.txt", "native.csv", "onemkl.csv")] +
                    [dict(path="report/source-hashes.txt")]))


def lab_index(root, rows):
    (root / "lab-cases.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (root / "lab-job.json").write_text(json.dumps(dict(id=42, status="DONE", experiment=dict(
        configs=[dict(label=arm) for arm in dict.fromkeys(row["arm"] for row in rows)],
        bench=dict(request=dict(parameters={}))))))
    (root / "lab-execution.json").write_text('[{"failedAttempt":{"cpuMillis":1234}}]')


class TuningCaptureTest(unittest.TestCase):
    def test_pairs_round_ratios_and_uses_runtime_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            capture = measurements([root], "native", "onemkl")
            self.assertEqual([0.5] * 3, capture.paired["dot+32+uniform"])
            self.assertEqual(10, len(capture.hashes))
            self.assertEqual(("Dot", 32, 1, 1, False, 1, True), capture.shapes["dot+32+uniform"])
            self.assertEqual(("native", "Linux", "X86_64", "measured CPU", "Avx2", "*"), capture.target("native"))
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
            capture = measurements([root], "native", "onemkl")
            with self.assertRaises(ValueError):
                capture.target("native")
            self.assertEqual("*", capture.target("native", cpu="*")[3])

    def test_vendor_tail_bound_uses_reported_cache_share_and_retains_known_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            path = root / "pass-1/metadata.txt"
            path.write_text(path.read_text().replace("flags=avx2", "flags=avx2\nl3 cache=80 MiB\ncore(s) per socket=20\nsocket(s)=2"))

            self.assertEqual(2 * 1024 ** 2, measurements([root], "native", "onemkl").cache_by_cpu["measured CPU"])
            path.write_text(path.read_text().replace("socket(s)=2", "socket(s)=-"))
            self.assertEqual(80 * 1024 ** 2, measurements([root], "native", "onemkl").cache_by_cpu["measured CPU"])

    def test_model_independent_profiles_require_a_single_verified_instruction_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            path = root / "pass-1/metadata.txt"
            path.write_text(path.read_text().replace("profile_cpu_class=Avx2", "profile_cpu_class=Avx512"))
            with self.assertRaises(ValueError):
                measurements([root], "native", "onemkl").target("native", cpu="*")
            for path in root.glob("*/metadata.txt"):
                path.write_text("\n".join("flags=" if line.startswith("flags=") else line for line in path.read_text().splitlines())
                                .replace("profile_cpu_class=Avx2", "profile_cpu_class=*").replace("profile_cpu_class=Avx512", "profile_cpu_class=*") + "\n")
            with self.assertRaises(ValueError):
                measurements([root], "native", "onemkl").target("native", cpu="*")

    def test_frozen_lab_placement_can_establish_the_instruction_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lab_index(root, [lab_row(0, "candidate")])
            rounds(root / "cases/0/out/report")
            for path in root.glob("cases/0/out/report/pass-*/metadata.txt"):
                path.write_text(path.read_text().replace("flags=avx2", "flags=").replace("profile_cpu_class=Avx2", "profile_cpu_class=*"))
            job = json.loads((root / "lab-job.json").read_text())
            job["experiment"]["nodeTags"] = dict(require=["simd:avx512"])
            (root / "lab-job.json").write_text(json.dumps(job))

            self.assertEqual("Avx512", measurements([root], "native", "onemkl").target("native", cpu="*")[4])

    def test_arm_flags_distinguish_scalable_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            for path in root.glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("profile_arch=X86_64", "profile_arch=Arm64")
                                .replace("profile_cpu_class=Avx2", "profile_cpu_class=*\nprofile_sve_vector_bits=256")
                                .replace("flags=avx2", "flags=fp asimd sve"))

            self.assertEqual("Sve", measurements([root], "native", "onemkl").target("native", cpu="*")[4])

    def test_supplemental_rows_retain_separate_clean_source_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("training", "supplement")]
            for root in roots:
                rounds(root)
            for path in roots[1].glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("a" * 40, "b" * 40))
            capture = measurements(roots, "native", "onemkl")

            self.assertEqual({"a" * 40, "b" * 40}, capture.commits)
            self.assertEqual([0.5] * 6, capture.paired["dot+32+uniform"])
            path = roots[1] / "pass-1/metadata.txt"
            path.write_text(path.read_text().replace("b" * 40, "c" * 40))
            with self.assertRaises(ValueError):
                measurements(roots, "native", "onemkl")

    def test_class_and_sve_width_projection_retains_excluded_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lab_index(root, [lab_row(i, "candidate", str(i)) for i in range(3)])
            for index, bits in enumerate((None, 128, 256)):
                report = root / f"cases/{index}/out/report"
                rounds(report, model=f"CPU {index}")
                if bits:
                    for path in report.glob("*/metadata.txt"):
                        path.write_text(path.read_text().replace("profile_arch=X86_64", "profile_arch=Arm64")
                                        .replace("profile_cpu_class=Avx2", f"profile_cpu_class=Sve\nprofile_sve_vector_bits={bits}"))

            capture = measurements([root], "native", "onemkl", cpu_class="Sve", sve_vector_bits=256)

            self.assertEqual(("Sve", "256"), capture.target("native", cpu="*")[4:])
            self.assertEqual({"CPU 2"}, capture.models)
            self.assertEqual(2, len(capture.excluded))
            self.assertEqual(3, len(capture.lab[0]["rows"]))
            self.assertEqual(set(), capture.missing_rows)
            with self.assertRaisesRegex(ValueError, "one verified vector width"):
                measurements([root], "native", "onemkl", cpu_class="Sve").target("native", cpu="*")
            for path in (root / "cases/2/out/report").glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("profile_sve_vector_bits=256", "profile_sve_vector_bits=*"))
            with self.assertRaisesRegex(ValueError, "unverified"):
                measurements([root], "native", "onemkl", cpu_class="Sve", sve_vector_bits=256)

    def test_inputs_cannot_mix_runtime_vendor_or_timing_protocols(self):
        for previous, replacement in (("native-version", "new-version"), ("harness=calibrated", "harness=other"),
                                      ("2026.1/threads", "2026.2/threads"), ("threads=1-confirmed", "threads=1-unconfirmed"),
                                      ("warmups=20", "warmups=21"), ("jvm_gc=serial", "jvm_gc=g1"),
                                      ("jvm_heap=512m", "jvm_heap=1g"), ("cpu_policy=unbound", "cpu_policy=spread")):
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as directory:
                roots = [Path(directory) / name for name in ("training", "supplement")]
                for root in roots:
                    rounds(root)
                for path in roots[1].glob("*/metadata.txt"):
                    path.write_text(path.read_text().replace(previous, replacement))
                with self.assertRaisesRegex(ValueError, "protocols"):
                    measurements(roots, "native", "onemkl")

    def test_changed_code_requires_agreeing_anchors_in_each_cpu_stratum(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("training", "supplement")]
            for root in roots:
                rounds(root)
            (roots[1] / "source-hashes.txt").write_text("b" * 64 + " koblas/src/commonMain/kotlin/PortableEngine.kt\n")
            with self.assertRaisesRegex(ValueError, "anchors"):
                measurements(roots, "native", "onemkl")
            for root in roots:
                for path in root.glob("*/*.csv"):
                    path.write_text(path.read_text() + path.read_text().splitlines()[1].replace("dot+32", "dot+64").replace("Dot:32:", "Dot:64:") + "\n")
            self.assertEqual(1, len(measurements(roots, "native", "onemkl").anchor_checks))
            for path in roots[1].glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("measured CPU", "another CPU"))
            with self.assertRaisesRegex(ValueError, "anchors"):
                measurements(roots, "native", "onemkl")
            for path in roots[1].glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("another CPU", "measured CPU"))
            for path in roots[1].glob("*/onemkl.csv"):
                lines = path.read_text().splitlines()
                row = lines[-1].split(',')
                row[-2] = str(float(row[-2]) * 3)
                path.write_text("\n".join([*lines[:-1], ','.join(row)]) + "\n")
            with self.assertRaisesRegex(ValueError, "anchors"):
                measurements(roots, "native", "onemkl")

    def test_anchors_cannot_certify_another_layout_or_contiguity(self):
        for borrowed, contiguous in ((True, True), (False, False)):
            with self.subTest(borrowed=borrowed, contiguous=contiguous):
                shapes = {"first": ("Gemm", 32, 32, 32, False, 1, True),
                          "second": ("Gemm", 64, 64, 64, False, 1, True),
                          "other": ("Gemm", 128, 128, 128, borrowed, 1, contiguous)}
                records = [dict(manifest=manifest, cpu="CPU", paired={case: [0.5] * 3 for case in shapes})
                           for manifest in ("before", "after")]

                with self.assertRaisesRegex(ValueError, "anchors"):
                    verify_anchors(records, shapes)

    def test_jvm_lab_options_and_gc_profiling_follow_the_selected_arm(self):
        for override in ({"jvm-options": "-XX:MaxVectorSize=32"}, {"profile-gc": "true"}):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                roots = [Path(directory) / name for name in ("training", "supplement")]
                for job_id, root in enumerate(roots):
                    root.mkdir()
                    row = lab_row(0, "candidate")
                    row["benchCase"] = dict(parameters={"jvm-options": "ignored"})
                    lab_index(root, [row])
                    rounds(root / "cases/0/out/report", runtime="jvm")
                    job = json.loads((root / "lab-job.json").read_text())
                    job["id"] = job_id
                    job["experiment"]["bench"]["request"]["parameters"] = {"jvm-options": "", "profile-gc": "false"}
                    job["experiment"]["configs"][0].update({"param." + key: value for key, value in override.items()} if job_id else {})
                    (root / "lab-job.json").write_text(json.dumps(job))

                capture = measurements([roots[0]], "native", "onemkl")
                self.assertEqual("", capture.provenance[0]["signature"]["protocol"]["JAVA_TOOL_OPTIONS"])
                with self.assertRaisesRegex(ValueError, "protocols"):
                    measurements(roots, "native", "onemkl")

    def test_pinned_package_is_required_only_for_the_selected_mkl_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            row = lab_row(0, "candidate")
            lab_index(root, [row])
            report = root / "cases/0/out/report"
            rounds(report)
            job = json.loads((root / "lab-job.json").read_text())
            job["experiment"]["bench"]["request"]["parameters"]["vendor"] = "onemkl"
            (root / "lab-job.json").write_text(json.dumps(job))

            with self.assertRaisesRegex(ValueError, "package"):
                measurements([root], "native", "onemkl")
            for path in report.glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("Linux", "Windows"))
            self.assertIsNone(measurements([root], "native", "onemkl").provenance[0]["signature"]["package"])
            for path in report.glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("Windows", "Linux"))
            package = report.parent / "vendor-package.txt"
            package.write_text("oneMKL 2026.1\nsha256=fixture\n")
            row["record"]["artifacts"].append(dict(path="vendor-package.txt"))
            (root / "lab-cases.jsonl").write_text(json.dumps(row) + "\n")
            self.assertEqual(package.read_text().strip(), measurements([root], "native", "onemkl").provenance[0]["signature"]["package"])
            for path in report.glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("/oneMKL/", "/OpenBLAS/"))
            package.unlink()
            self.assertIsNone(measurements([root], "native", "onemkl").provenance[0]["signature"]["package"])

    def test_duplicate_capture_paths_or_jobs_cannot_double_count_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("training", "duplicate")]
            for root in roots:
                root.mkdir()
                lab_index(root, [lab_row(0, "candidate")])
                rounds(root / "cases/0/out/report")
            for inputs in ([roots[0], roots[0]], roots):
                with self.assertRaisesRegex(ValueError, "repeated"):
                    measurements(inputs, "native", "onemkl")

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
                    measurements([root], "native", "onemkl")

    def test_a_declared_operation_cannot_disappear_when_a_training_point_is_untimed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            path = root / "pass-2/onemkl.csv"
            path.write_text(path.read_text().replace(",ok,", ",unsupported,"))
            with self.assertRaisesRegex(ValueError, "untimed"):
                measurements([root], "native", "onemkl", backend="onemkl")

    def test_validation_rejects_empty_untimed_or_inconsistent_cases(self):
        for replacement in ("", "dot+32+uniform,unsupported,arithmetic,Dot:32:1:1:false:1:true,,onemkl\n",
                            "dot+64+uniform,ok,arithmetic,Dot:64:1:1:false:1:true,5,onemkl\n"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rounds(root)
                for path in root.glob("*/onemkl.csv"):
                    path.write_text(path.read_text().splitlines()[0] + "\n" + replacement)
                with self.assertRaises(ValueError):
                    measurements([root], "native", "onemkl", strict=True)

    def test_accepts_only_complete_rows_and_retains_failure_cost_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [lab_row(0, "base"), lab_row(1, "candidate"),
                    lab_row(2, "base", "broken"), lab_row(3, "candidate", "broken", "FAILED")]
            lab_index(root, rows)
            rounds(root / "cases/1/out/report")
            # Tempting timings in an excluded row must never enter a fitted crossover.
            rounds(root / "cases/3/out/report", "another CPU")
            capture = measurements([root], "native", "onemkl", arm="candidate")
            self.assertEqual([1], capture.lab[0]["accepted_rows"])
            self.assertEqual({"measured CPU"}, capture.models)
            self.assertEqual(4, len(capture.lab[0]["rows"]))
            self.assertEqual(1234, capture.lab[0]["resources"]["lab-execution.json"][0]["failedAttempt"]["cpuMillis"])
            self.assertEqual({("broken", None, 0)}, capture.missing_rows)
            with self.assertRaisesRegex(ValueError, "requested rows"):
                measurements([root], "native", "onemkl", arm="candidate", strict=True)
            with self.assertRaises(ValueError):
                measurements([root], "native", "onemkl")

    def test_missing_arm_and_escaping_artifacts_cannot_supply_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [lab_row(0, "base"), lab_row(1, "candidate", "another problem")]
            lab_index(root, rows)
            with self.assertRaises(ValueError):
                measurements([root], "native", "onemkl", arm="candidate")
            rows[1]["problem"]["problem"] = "dot"
            rows[1]["record"]["artifacts"][0]["path"] = "../../../../outside/pass-1/metadata.txt"
            lab_index(root, rows)
            with self.assertRaises(ValueError):
                measurements([root], "native", "onemkl", arm="candidate")

    def test_different_cpu_models_stay_in_separate_strata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [lab_row(0, "candidate"), lab_row(1, "candidate", "dot repetition")]
            lab_index(root, rows)
            rounds(root / "cases/0/out/report", "CPU A")
            rounds(root / "cases/1/out/report", "CPU B")
            for path in (root / "cases/1/out/report").glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("6.12 test", "6.13 other").replace("native-version", "another-runtime-version")
                                .replace("2026.1/threads", "2026.2/threads"))
            capture = measurements([root], "native", "onemkl")
            self.assertEqual({"CPU A": [0.5] * 3, "CPU B": [0.5] * 3}, capture.by_cpu["dot+32+uniform"])
            with self.assertRaises(ValueError):
                capture.target("native")
            self.assertEqual("*", capture.target("native", cpu="*")[3])
            for path in (root / "cases/1/out/report").glob("*/metadata.txt"):
                path.write_text(path.read_text().replace("CPU B", "CPU A"))
            with self.assertRaisesRegex(ValueError, "within a CPU model"):
                measurements([root], "native", "onemkl")

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
            capture = measurements([root], "jvm-owned", "onemkl", owned_target="jvm-default")
            self.assertEqual([0.5, 2.0, 0.5], capture.paired["dot+32+uniform"])

    def test_capture_rounds_cannot_mix_source_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rounds(root)
            path = root / "pass-3/metadata.txt"
            path.write_text(path.read_text().replace("a" * 40, "b" * 40))
            with self.assertRaises(ValueError):
                measurements([root], "native", "onemkl")


if __name__ == "__main__":
    unittest.main()
