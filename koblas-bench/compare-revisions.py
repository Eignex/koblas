#!/usr/bin/env python3
"""Capture a pinned JVM baseline, this JVM revision and native oneMKL on one worker."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import urllib.request
import zipfile
import shlex

ROOT = Path(__file__).resolve().parent.parent
MKL_PACKAGE = "intel-oneapi-mkl-core-2026.1-2026.1.0-236_amd64.deb"
JDK27_URL = "https://download.oracle.com/java/27/latest/jdk-27_linux-x64_bin.tar.gz"
JDK27_SHA256 = "d41bdc29a13920e90ee0f8e97a5bb9cb854ac06e0aa648a7fda1620db6c4e1ee"
MKL_SHA256 = "3b230c02fc0ac44f8f71062c20cd883b37c6c1e00940dc0a7f13b02d9c47bd93"


def run(command, directory, environment, log):
    with log.open("w") as stream:
        subprocess.run(command, cwd=directory, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def fingerprint(tree, binaries=False, output=None):
    result = {}
    if binaries:
        directories = ["koblas/build/libs", "koblas-bench/build/classes", "koblas-bench/build/bin", "koblas-bench/jmh/build/libs"]
        paths = [path for directory in directories for path in (tree / directory).rglob("*") if path.is_file()]
    else:
        paths = []
        excluded = {".git", ".gradle", ".kotlin", ".idea", "build", "node_modules", "__pycache__"}
        for directory, subdirectories, files in os.walk(tree):
            subdirectories[:] = [name for name in subdirectories if name not in excluded
                                 and Path(directory, name).resolve() != output]
            paths.extend(Path(directory, name) for name in files if name != ".git")
    for path in sorted(paths):
        result[str(path.relative_to(tree))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=ROOT / "koblas-bench/cases-reductions.txt")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--target-ms", type=int, default=200)
    parser.add_argument("--forks", type=int, default=2)
    parser.add_argument("--cpu", default="auto")
    parser.add_argument("--jvm-gc", choices=["g1", "serial", "zgc"])
    parser.add_argument("--jvm-heap", help="Use the same fixed JVM benchmark heap on both revisions, for example 1g")
    parser.add_argument("--install-onemkl", action="store_true")
    parser.add_argument("--build-heap", help="Bound each Gradle build heap, for example 2g, and compile Kotlin in process")
    parser.add_argument("--baseline-java-version", type=int, choices=[25, 27])
    parser.add_argument("--candidate-java-version", type=int, choices=[25, 27])
    parser.add_argument("--install-jdk27", action="store_true")
    parser.add_argument("--require-same-source", action="store_true")
    parser.add_argument("--runtime-conformance", action="store_true")
    args = parser.parse_args()
    versions = {"baseline": args.baseline_java_version, "candidate": args.candidate_java_version}
    if (args.baseline_java_version is None) != (args.candidate_java_version is None):
        parser.error("Specify both runtime versions together")
    if args.install_jdk27 and 27 not in versions.values():
        parser.error("Installing JDK27 requires a selected JDK27 arm")
    if not re.fullmatch(r"[0-9a-f]{40}", args.baseline_ref):
        parser.error("baseline-ref must be a full commit SHA")
    if args.rounds < 3 or args.samples < 1 or args.warmups < 1 or args.target_ms < 1 or args.forks < 1:
        parser.error("require at least three rounds and positive measurement settings")
    if args.build_heap is not None and not re.fullmatch(r"[1-9][0-9]*[mg]", args.build_heap):
        parser.error("build-heap must be a positive size ending in m or g")
    if args.jvm_heap is not None and not re.fullmatch(r"[1-9][0-9]*[mg]", args.jvm_heap):
        parser.error("jvm-heap must be a positive size ending in m or g")
    if args.runtime_conformance and not args.install_jdk27:
        parser.error("Runtime conformance requires the installed pinned JDK27")
    cpu = min(os.sched_getaffinity(0)) if args.cpu == "auto" else int(args.cpu)
    if cpu not in os.sched_getaffinity(0):
        parser.error("cpu is outside the allowed affinity")
    cases = args.cases.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases.txt").write_bytes(cases.read_bytes())
    plan = dict(status="incomplete", baseline_ref=args.baseline_ref, cpu=cpu, rounds=args.rounds,
                samples=args.samples, warmups=args.warmups, target_ms=args.target_ms, forks=args.forks,
                aggregation="median of per-round medians", orders=[])
    plan["jvm_gc"] = args.jvm_gc or "default"
    plan["jvm_heap"] = args.jvm_heap or "default"
    plan_path = output / "comparison.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    environment = os.environ.copy()
    if args.build_heap is not None:
        # A separate Kotlin daemon can otherwise outlive a build and share the worker's hard memory cap.
        environment["GRADLE_OPTS"] = environment.get("GRADLE_OPTS", "") + " -Dorg.gradle.project.kotlin.compiler.execution.strategy=in-process"
        environment["GRADLE_OPTS"] = environment.get("GRADLE_OPTS", "") + \
            f" -Dorg.gradle.jvmargs=-Xmx{args.build_heap} -Dorg.gradle.workers.max=2"
        plan["build_heap"] = args.build_heap
        plan["kotlin_compiler_execution_strategy"] = "in-process"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    scratch_parent = ROOT / "koblas-bench/build"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="revision-comparison-", dir=scratch_parent) as temporary:
        scratch = Path(temporary)
        runtime_home = None
        if args.install_jdk27:
            archive = scratch / "jdk27.tar.gz"
            urllib.request.urlretrieve(JDK27_URL, archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != JDK27_SHA256:
                raise RuntimeError("JDK27 archive checksum differs from the validated runtime")
            runtime_directory = scratch / "jdk27"
            runtime_directory.mkdir()
            run(["tar", "--extract", "--gzip", "--no-same-owner", "--file", str(archive), "--directory", str(runtime_directory)],
                ROOT, environment, output / "jdk27-extract.log")
            homes = [p.parent.parent for p in runtime_directory.glob("*/bin/java")]
            if len(homes) != 1:
                raise RuntimeError("JDK27 archive has an unexpected layout")
            runtime_home = homes[0]
            plan["jdk27_archive"] = dict(url=JDK27_URL, sha256=JDK27_SHA256,
                                         java_sha256=hashlib.sha256((runtime_home / "bin/java").read_bytes()).hexdigest(),
                                         release=(runtime_home / "release").read_text())

        def runtime_environment(arm):
            result = dict(environment)
            if versions.get(arm) is not None:
                properties = {"bench.javaVersion": str(versions[arm])}
                if versions[arm] == 27 and runtime_home is not None:
                    properties["bench.expectedJavaHome"] = str(runtime_home)
                # Shell launchers discard environment entries containing dots; system properties survive them.
                result["GRADLE_OPTS"] = result.get("GRADLE_OPTS", "") + " " + " ".join(
                    shlex.quote("-Dorg.gradle.project." + name + "=" + value) for name, value in properties.items())
            return result
        if args.install_onemkl:
            package = scratch / MKL_PACKAGE
            urllib.request.urlretrieve("https://apt.repos.intel.com/oneapi/pool/main/" + MKL_PACKAGE, package)
            if hashlib.sha256(package.read_bytes()).hexdigest() != MKL_SHA256:
                raise RuntimeError("oneMKL package checksum differs from the pinned reference container")
            vendor = scratch / "vendors"
            run(["dpkg-deb", "--extract", str(package), str(vendor)], ROOT, environment, output / "vendor-extract.log")
            mkl = vendor / "opt/intel/oneapi/mkl/2026.1"
            libraries = mkl / "lib"
            if not (libraries / "libmkl_rt.so.3").is_file():
                raise RuntimeError("oneMKL package did not contain the expected runtime")
            environment["MKLROOT"] = str(mkl)
            environment["LD_LIBRARY_PATH"] = str(libraries) + ":" + environment.get("LD_LIBRARY_PATH", "")
            environment["MKL_THREADING_LAYER"] = "SEQUENTIAL"
            plan["onemkl_package"] = dict(file=MKL_PACKAGE, sha256=MKL_SHA256)
        baseline = scratch / "baseline"
        baseline.mkdir()
        for index, command in enumerate([
            ["git", "init", "--quiet"],
            ["git", "remote", "add", "origin", "https://github.com/Eignex/koblas.git"],
            ["git", "fetch", "--depth=1", "origin", args.baseline_ref],
            ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        ]):
            run(command, baseline, environment, output / f"baseline-source-{index}.log")
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=baseline, text=True).strip()
        if actual != args.baseline_ref:
            raise RuntimeError("fetched baseline does not match the requested revision")
        # One capture driver lets older kernels run the same measurement protocol without editing them.
        driver = ROOT / "koblas-bench/capture-report.sh"
        plan["capture_driver_sha256"] = hashlib.sha256(driver.read_bytes()).hexdigest()
        for name, tree in [("baseline", baseline), ("candidate", ROOT)]:
            tasks = [":koblas-bench:jvmMainClasses"]
            if name == "candidate":
                tasks.append(":koblas-bench:linkReleaseExecutableLinuxX64")
            run(["./gradlew", "--no-daemon", *tasks], tree, environment, output / f"{name}-build.log")
        trees = {"baseline": baseline, "candidate": ROOT}
        if args.baseline_java_version is not None:
            plan["runtime_versions"] = versions
            for arm, tree in trees.items():
                run(["./gradlew", "--no-daemon", ":koblas-bench:prepareBenchmarkRuntime"], tree,
                    runtime_environment(arm), output / (arm + "-runtime.log"))
        if args.runtime_conformance:
            checks = [":koblas:simdG1AlignmentCheck", ":koblas:simdSerialAlignmentCheck", ":koblas:simdObjectAlignmentCheck"]
            run(["./gradlew", "--no-daemon", *checks, "-Pkoblas.runtimeCheckJavaHome=" + str(runtime_home)],
                ROOT, environment, output / "jdk27-reduction-conformance.log")
            checked_runtime = (output / "jdk27-reduction-conformance.log").read_text()
            if len(re.findall(r"reduction alignment conformance passed java=27\.", checked_runtime)) != len(checks):
                raise RuntimeError("Reduction conformance did not execute on JDK27")
            plan["runtime_conformance"] = dict(java=27, tasks=checks, status="passed")
        if args.require_same_source:
            if fingerprint(baseline, output=output) != fingerprint(ROOT, output=output):
                raise RuntimeError("Runtime comparison requires identical source trees")
            jars = [next((tree / "koblas/build/libs").glob("koblas-jvm-SNAPSHOT.jar")) for tree in trees.values()]
            with zipfile.ZipFile(jars[0]) as first, zipfile.ZipFile(jars[1]) as second:
                names = {n for n in first.namelist() if not n.endswith("/")}
                if names != {n for n in second.namelist() if not n.endswith("/")} or any(first.read(n) != second.read(n) for n in names):
                    raise RuntimeError("Runtime comparison library contents differ")
            plan["identical_sources_and_library_contents"] = True
        hashes = {name: {"sources": fingerprint(tree, output=output), "binaries": fingerprint(tree, binaries=True)}
                  for name, tree in trees.items()}
        (output / "input-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
        arms = ["baseline", "candidate", "onemkl"]
        observations = {}
        for round_index in range(args.rounds):
            order = arms[round_index % 3:] + arms[:round_index % 3]
            plan["orders"].append(order)
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
            for arm in order:
                print(f"round {round_index + 1}: {arm}", flush=True)
                report = output / f"round-{round_index + 1}-{arm}"
                tree = baseline if arm == "baseline" else ROOT
                target = "onemkl" if arm == "onemkl" else "jvm-simd"
                command = [str(driver), "--libraries", "onemkl",
                           "--targets", target, "--cases", str(output / "cases.txt"), "--cpu", str(cpu),
                           "--samples", str(args.samples), "--warmups", str(args.warmups),
                           "--target-ms", str(args.target_ms), "--forks", str(args.forks),
                           "--output", str(report)]
                if args.jvm_gc is not None:
                    command += ["--jvm-gc", args.jvm_gc]
                if args.jvm_heap is not None:
                    command += ["--jvm-heap", args.jvm_heap]
                capture_environment = dict(runtime_environment(arm), KOBLAS_CAPTURE_ROOT=str(tree))
                run(command, tree, capture_environment, output / f"round-{round_index + 1}-{arm}.log")
                for name, checked_tree in trees.items():
                    if fingerprint(checked_tree, output=output) != hashes[name]["sources"] or \
                            fingerprint(checked_tree, binaries=True) != hashes[name]["binaries"]:
                        raise RuntimeError(f"{name} sources or runtime inputs changed during the comparison")
                with (report / f"{target}.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != len([line for line in (output / "cases.txt").read_text().splitlines()
                                     if line.strip() and not line.lstrip().startswith("#")]):
                    raise RuntimeError("capture did not return every requested case")
                if versions.get(arm) is not None:
                    fork_lines = [line[line.index("fork_runtime "):] for line in (output / f"round-{round_index + 1}-{arm}.log").read_text().splitlines()
                                  if "fork_runtime " in line]
                    if versions[arm] == 27 and runtime_home is not None:
                        invokers = re.findall(r"# VM invoker: (.+)", (output / f"round-{round_index + 1}-{arm}.log").read_text())
                        if len(invokers) != len(rows) or any(Path(path).resolve() != (runtime_home / "bin/java").resolve() for path in invokers):
                            raise RuntimeError("Measured forks did not use the pinned JDK27 executable")
                    expected_forks = len(rows) * args.forks
                    if len(fork_lines) != expected_forks or any(f" java={versions[arm]}." not in line for line in fork_lines):
                        raise RuntimeError("Measured forks do not match the requested runtime")
                for row in rows:
                    if row["status"] != "ok" or not float(row["median_ns"]) > 0:
                        raise RuntimeError(f"capture did not measure {arm}: {row}")
                    observations.setdefault((arm, row["case"]), []).append(float(row["median_ns"]))
                # Retain completed observations even when a later capture fails.
                (output / "observations.json").write_text(json.dumps(
                    [dict(arm=key[0], case=key[1], round_medians_ns=value)
                     for key, value in observations.items()], indent=2) + "\n")
        with (output / "summary.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case", "arm", "median_ns", "min_round_ns", "max_round_ns"])
            for (arm, case), values in observations.items():
                writer.writerow([case, arm, statistics.median(values), min(values), max(values)])
        plan["status"] = "complete"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
