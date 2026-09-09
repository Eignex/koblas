#!/usr/bin/env python3
"""Create and inspect portable koblas hardware benchmark bundles."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from hardware_bundle import (  # noqa: E402
    BUNDLE_SCHEMA,
    FAILURE_PATTERN,
    ReportError,
    aggregate_rows,
    bundle_directory,
    check_pass,
    checksums,
    load_pass,
    summary_text,
    validate_directory,
    write_rows,
)


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "koblas-bench"
CATALOG_PATH = BENCH / "hardware-workload-v1.json"
CAPABILITIES = {
    "jvm": {"built-in": "supported", "openblas": "supported", "onemkl": "supported"},
    "linuxX64": {"built-in": "supported", "openblas": "supported", "onemkl": "unsupported"},
    "macosArm64": {"built-in": "supported", "openblas": "unsupported", "onemkl": "unsupported"},
}
COMPARATOR_LIBRARY_CANDIDATES = {
    "openblas": ["libopenblas.so.0", "libopenblas.so", "libopenblas.dylib"],
    "onemkl": [
        "libmkl_rt.so.3",
        "libmkl_rt.so.2",
        "libmkl_rt.so",
        "libmkl_rt.3.dylib",
        "libmkl_rt.dylib",
        "mkl_rt.3.dll",
        "mkl_rt.2.dll",
        "mkl_rt.dll",
    ],
}


def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text())


def normalize_target(value: str) -> str:
    if value == "native":
        key = (platform.system(), platform.machine().lower())
        if key in {("Linux", "x86_64"), ("Linux", "amd64")}:
            return "linuxX64"
        if key == ("Darwin", "arm64"):
            return "macosArm64"
        raise ReportError("native reports support Linux x86-64 and Apple Silicon hosts")
    if value not in CAPABILITIES:
        raise ReportError("target must be jvm, native, linuxX64, or macosArm64")
    return value


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    log: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    chunks: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        chunks.append(line)
        sys.stderr.write(line)
    return_code = process.wait()
    output_text = "".join(chunks)
    if log is not None:
        log.write_text(output_text)
    return subprocess.CompletedProcess(command, return_code, output_text, "")


def output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def probe_library(comparator: str) -> dict[str, str]:
    candidates = COMPARATOR_LIBRARY_CANDIDATES[comparator]
    symbol = "openblas_get_config" if comparator == "openblas" else "MKL_Get_Version_String"
    code = r'''
import ctypes, json, sys
names, symbol = json.loads(sys.argv[1]), sys.argv[2]
for name in names:
    try:
        lib = ctypes.CDLL(name)
        fn = getattr(lib, symbol)
        if symbol == "openblas_get_config":
            fn.restype = ctypes.c_char_p
            version = (fn() or b"unknown").decode(errors="replace")
        else:
            buffer = ctypes.create_string_buffer(256)
            fn(buffer, len(buffer))
            version = buffer.value.decode(errors="replace") or "unknown"
        print(json.dumps({"availability":"available", "library":name, "version":version}))
        raise SystemExit(0)
    except (OSError, AttributeError):
        pass
print(json.dumps({"availability":"unavailable", "library":"unknown", "version":"unknown"}))
'''
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(candidates), symbol],
        text=True,
        capture_output=True,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"availability": "unavailable", "library": "unknown", "version": "unknown"}


def comparator_status(target: str) -> dict[str, dict[str, str]]:
    status: dict[str, dict[str, str]] = {
        "built-in": {"capability": "supported", "availability": "available", "version": "koblas"}
    }
    for comparator in ("openblas", "onemkl"):
        capability = CAPABILITIES[target][comparator]
        if capability == "unsupported":
            status[comparator] = {
                "capability": capability,
                "availability": "unsupported",
                "version": "unknown",
            }
        else:
            status[comparator] = {"capability": capability, **probe_library(comparator)}
    return status


def parse_comparators(request: str, status: dict[str, dict[str, str]]) -> list[str]:
    if request == "none":
        return []
    if request == "auto":
        return [name for name in ("openblas", "onemkl") if status[name]["availability"] == "available"]
    requested = [part.strip().lower() for part in request.split(",") if part.strip()]
    unknown = sorted(set(requested) - {"openblas", "onemkl"})
    if unknown:
        raise ReportError(f"unknown comparator(s): {', '.join(unknown)}")
    for name in requested:
        value = status[name]["availability"]
        if value != "available":
            raise ReportError(f"explicitly requested comparator {name} is {value} for this target")
    return list(dict.fromkeys(requested))


def cpu_model() -> str:
    if platform.system() == "Linux":
        value = output(["lscpu"])
        match = re.search(r"^Model name:\s*(.+)$", value, re.MULTILINE)
        return match.group(1).strip() if match else "unknown"
    if platform.system() == "Darwin":
        return output(["sysctl", "-n", "machdep.cpu.brand_string"])
    return "unknown"


def jvm_metadata() -> dict[str, str]:
    result = subprocess.run(
        [str(ROOT / "gradlew"), "-q", ":koblas-bench:benchmarkJvmMetadata"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return {"executable": "unknown", "version": "unknown", "vendor": "unknown", "runtime": "unknown"}
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def native_metadata() -> dict[str, str]:
    resolved = subprocess.run(
        [str(ROOT / "gradlew"), "-q", ":koblas-bench:benchmarkNativeMetadata"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if resolved.returncode != 0:
        return {
            "kind": "Kotlin/Native",
            "compiler": "unknown",
            "distribution": "unknown",
            "runtime": "native executable",
        }
    metadata = dict(line.split("=", 1) for line in resolved.stdout.splitlines() if "=" in line)
    executable = metadata.pop("executable", "")
    if not executable:
        return {
            "kind": "Kotlin/Native",
            "compiler": "unknown",
            "distribution": metadata.get("distribution", "unknown"),
            "runtime": "native executable",
        }
    try:
        result = subprocess.run([executable, "-version"], cwd=ROOT, text=True, capture_output=True)
        version = (result.stdout + result.stderr).strip().replace("\n", " ") if result.returncode == 0 else "unknown"
    except OSError:
        version = "unknown"
    return {**metadata, "compiler": version}


def preflight_data(target: str) -> dict[str, Any]:
    gradle_text = output([str(ROOT / "gradlew"), "--version", "--quiet"])
    return {
        "target": target,
        "host": {
            "os": platform.system() or "unknown",
            "os_release": platform.release() or "unknown",
            "architecture": platform.machine() or "unknown",
            "cpu_model": cpu_model(),
            "logical_cpus": os.cpu_count() if os.cpu_count() is not None else "unknown",
        },
        "gradle": gradle_text.splitlines()[0] if gradle_text != "unknown" else "unknown",
        "runtime": jvm_metadata() if target == "jvm" else native_metadata(),
        "comparators": comparator_status(target),
    }


def print_preflight(data: dict[str, Any]) -> None:
    print(f"target: {data['target']}")
    print(f"host: {data['host']['os']} {data['host']['architecture']} ({data['host']['cpu_model']})")
    runtime = data["runtime"]
    runtime_name = runtime.get("executable") or runtime.get("compiler") or runtime.get("kind", "unknown")
    print(f"benchmark runtime: {runtime_name} {runtime.get('runtime', 'unknown')}")
    print("comparators:")
    for name, status in data["comparators"].items():
        print(f"  {name}: {status['availability']} ({status.get('version', 'unknown')})")


def task_name(target: str, smoke: bool) -> str:
    configuration = "HardwareSmoke" if smoke else "Hardware"
    return f":koblas-bench:{target}{configuration}Benchmark"


def build_task(target: str) -> str:
    if target == "jvm":
        return ":koblas-bench:jvmBenchmarkCompile"
    cap = target[0].upper() + target[1:]
    return f":koblas-bench:link{cap}BenchmarkReleaseExecutable{cap}"


def gradle_command(args: list[str], cores: str | None) -> list[str]:
    command = [str(ROOT / "gradlew"), *args, "--no-daemon"]
    if cores and platform.system() == "Linux" and shutil.which("taskset"):
        return ["taskset", "-c", cores, *command]
    return command


def find_result(report_root: Path) -> Path:
    results = sorted(report_root.rglob("*.json")) if report_root.exists() else []
    if len(results) != 1:
        raise ReportError(f"expected exactly one result JSON in invocation directory, found {len(results)}")
    return results[0]


def command_validate(path: Path, summarize: bool) -> None:
    with bundle_directory(path) as directory:
        metadata, rows = validate_directory(directory)
        print(f"valid koblas hardware bundle: {metadata['run_id']} ({len(rows)} rows)")
        if summarize:
            print(summary_text(metadata, rows), end="")


def execute_report(args: argparse.Namespace, smoke: bool) -> Path:
    target = normalize_target(args.target)
    catalog = load_catalog()
    preflight = preflight_data(target)
    comparators = parse_comparators(args.comparators, preflight["comparators"])
    arms = ["built-in", *comparators]
    mode = "smoke" if smoke else "standard"
    expected = {arm: catalog["expected_cases"][mode][arm] for arm in arms}
    print_preflight(preflight)
    print(f"workload: {sum(expected.values())} cases/pass across {', '.join(arms)}; 2 fresh passes")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    output_root = Path(args.output).expanduser().resolve()
    stage = output_root / run_id
    raw_root = stage / "raw"
    stage.mkdir(parents=True, exist_ok=False)
    active_root = BENCH / "build" / "hardware-active"
    active_root.mkdir(parents=True, exist_ok=True)
    marker = active_root / run_id
    marker.write_text(str(os.getpid()))
    concurrent_seen: set[str] = set()
    started = time.monotonic()
    pass_rows: list[dict[str, Any]] = []
    execution_ids: list[str] = []
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    for assignment in args.tuning:
        if "=" not in assignment:
            raise ReportError(f"tuning override must be NAME=VALUE: {assignment}")
        name, value = assignment.split("=", 1)
        if not name.startswith("KOBLAS_"):
            raise ReportError("tuning override names must start with KOBLAS_")
        environment[name] = value
    try:
        result = run_command(gradle_command([build_task(target)], args.cores), env=environment, log=stage / "build.log")
        if result.returncode != 0:
            raise ReportError("benchmark build failed; preserved invocation directory contains build.log")
        for pass_number in (1, 2):
            for arm in arms:
                concurrent_seen.update(item.name for item in active_root.iterdir() if item.is_file() and item.name != run_id)
                execution_id = f"{arm}-pass-{pass_number}-{uuid.uuid4().hex[:8]}"
                execution_ids.append(execution_id)
                raw_dir = raw_root / arm
                raw_dir.mkdir(parents=True, exist_ok=True)
                relative_reports = f"hardware-runs/{run_id}/gradle/{execution_id}"
                relative_descriptions = f"hardware-runs/{run_id}/descriptions/{execution_id}"
                command = gradle_command([
                    task_name(target, smoke),
                    f"-Pbench.hardwareComparator={arm}",
                    f"-Pbench.reportsDir={relative_reports}",
                    f"-Pbench.descriptionsDir={relative_descriptions}",
                ], args.cores)
                log_path = raw_dir / f"pass-{pass_number}.log"
                result = run_command(command, env=environment, log=log_path)
                if result.returncode != 0 or FAILURE_PATTERN.search(result.stdout):
                    raise ReportError(f"benchmark fork failed for {arm} pass {pass_number}; raw log preserved")
                generated = find_result(BENCH / "build" / relative_reports)
                destination = raw_dir / f"pass-{pass_number}.json"
                shutil.copy2(generated, destination)
                rows = load_pass(destination, arm, pass_number, catalog["profile_version"])
                ids = check_pass(rows, arm, expected[arm])
                previous = {row["case_id"] for row in pass_rows if row["arm"] == arm and row["pass"] == 1}
                if pass_number == 2 and ids != previous:
                    raise ReportError(f"{arm} passes do not contain identical case IDs")
                pass_rows.extend(rows)

        aggregates = aggregate_rows(pass_rows)
        settings = catalog["settings"] if not smoke else {
            **catalog["settings"],
            "warmups": 1,
            "iterations": 1,
            "iteration_time_ms": 20,
        }
        tuning = {key: environment[key] for key in sorted(environment) if key.startswith("KOBLAS_")}
        resolved_lines = []
        runtime_lines = []
        for log in raw_root.rglob("*.log"):
            for line in log.read_text(errors="replace").splitlines():
                resolved_lines.extend(match.group(0) for match in re.finditer(r"resolved:[^\r\n]+", line))
                if line.startswith("# VM version:") or line.startswith("# VM invoker:"):
                    runtime_lines.append(line)
        metadata = {
            "schema_version": BUNDLE_SCHEMA,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "target": target,
            "arms": arms,
            "expected_cases": expected,
            "execution_ids": execution_ids,
            "repository": {
                "commit": output(["git", "rev-parse", "HEAD"]),
                "dirty": output(["git", "status", "--porcelain"]) not in {"", "unknown"},
            },
            "workload": {
                "profile": catalog["profile"],
                "version": catalog["profile_version"],
                "seed": catalog["seed"],
                "settings": settings,
            },
            "host": preflight["host"],
            "runtime": {
                **preflight["runtime"],
                "measured_process_lines": sorted(set(runtime_lines)) or ["unknown"],
            },
            "implementations": sorted(set(resolved_lines)) or ["unknown"],
            "comparators": preflight["comparators"],
            "threading": {
                "benchmark_threads": 1,
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
            "affinity": f"taskset -c {args.cores}" if args.cores else "none",
            "tuning_overrides": tuning,
            "quality": {
                "concurrent_runs_detected": sorted(concurrent_seen),
                "annotation": "concurrent report activity detected; retain uncertainty and raw passes"
                if concurrent_seen
                else "no concurrent koblas report runner detected; other machine activity was not excluded",
            },
            "duration_seconds": round(time.monotonic() - started, 3),
            "privacy": "Structured metadata omits hostname, username, and checkout path. Logs may contain local paths; inspect before sharing.",
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        shutil.copy2(CATALOG_PATH, stage / CATALOG_PATH.name)
        shutil.copy2(BENCH / "benchmark-coverage.tsv", stage / "benchmark-coverage.tsv")
        shutil.copy2(BENCH / "comparator-coverage.tsv", stage / "comparator-coverage.tsv")
        write_rows(stage, aggregates)
        (stage / "summary.txt").write_text(summary_text(metadata, aggregates))
        checksums(stage)
        validate_directory(stage)
        archive = output_root / f"koblas-hardware-{target}-{run_id}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(stage, arcname=stage.name)
        print(f"completed in {metadata['duration_seconds']:.1f}s; {len(aggregates)} machine-readable rows")
        print(archive)
        return archive
    finally:
        marker.unlink(missing_ok=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="probe the host without running benchmarks")
    preflight.add_argument("target", nargs="?", default="jvm")
    for name in ("smoke", "standard"):
        run = subparsers.add_parser(name, help=f"run the {name} hardware workload")
        run.add_argument("target", nargs="?", default="jvm")
        run.add_argument("--comparators", default="none", help="none, auto, openblas, onemkl, or a comma-separated list")
        run.add_argument("--output", default=str(BENCH / "build" / "reports" / "hardware"))
        run.add_argument("--cores", help="Linux CPU list passed to taskset, for example 2-5")
        run.add_argument("--tuning", action="append", default=[], metavar="KOBLAS_NAME=VALUE")
    validate = subparsers.add_parser("validate", help="validate an unpacked bundle or tar.gz offline")
    validate.add_argument("bundle", type=Path)
    summarize = subparsers.add_parser("summarize", help="validate and summarize a bundle offline")
    summarize.add_argument("bundle", type=Path)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        if args.command == "preflight":
            print_preflight(preflight_data(normalize_target(args.target)))
        elif args.command in {"smoke", "standard"}:
            execute_report(args, smoke=args.command == "smoke")
        else:
            command_validate(args.bundle.resolve(), summarize=args.command == "summarize")
        return 0
    except (ReportError, OSError, json.JSONDecodeError, tarfile.TarError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
