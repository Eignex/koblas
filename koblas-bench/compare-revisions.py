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

ROOT = Path(__file__).resolve().parent.parent
MKL_PACKAGE = "intel-oneapi-mkl-core-2026.1-2026.1.0-236_amd64.deb"
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
    parser.add_argument("--install-onemkl", action="store_true")
    parser.add_argument("--build-heap", help="Bound each Gradle build heap, for example 2g, and compile Kotlin in process")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.baseline_ref):
        parser.error("baseline-ref must be a full commit SHA")
    if args.rounds < 3 or args.samples < 1 or args.warmups < 1 or args.target_ms < 1 or args.forks < 1:
        parser.error("require at least three rounds and positive measurement settings")
    if args.build_heap is not None and not re.fullmatch(r"[1-9][0-9]*[mg]", args.build_heap):
        parser.error("build-heap must be a positive size ending in m or g")
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
                capture_environment = dict(environment, KOBLAS_CAPTURE_ROOT=str(tree))
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
