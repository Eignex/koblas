#!/usr/bin/env python3
"""Create and validate portable koblas hardware benchmark bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_SCHEMA = 1
ROW_SCHEMA = 1
ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "koblas-bench"
CATALOG_PATH = BENCH / "hardware-workload-v1.json"
FAILURE_PATTERN = re.compile(r"<failure>|EXCEPTION: <ERROR>|Benchmark fork reported", re.IGNORECASE)
ARM_PARAMETERS = {"denseArm", "sparseArm", "comparator"}
V1_SPARSE_PRODUCT_BENCHMARK = "com.eignex.koblas.bench.SparseProductHostBenchmark"
V1_TRIANGLE_VARIANT = "upper-nontrans-nonunit"
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


class ReportError(RuntimeError):
    pass


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


def run_command(command: list[str], *, env: dict[str, str] | None = None, log: Path | None = None) -> subprocess.CompletedProcess[str]:
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
            status[comparator] = {"capability": capability, "availability": "unsupported", "version": "unknown"}
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
        return {"kind": "Kotlin/Native", "compiler": "unknown", "distribution": "unknown", "runtime": "native executable"}
    metadata = dict(line.split("=", 1) for line in resolved.stdout.splitlines() if "=" in line)
    executable = metadata.pop("executable", "")
    if not executable:
        return {"kind": "Kotlin/Native", "compiler": "unknown", "distribution": metadata.get("distribution", "unknown"), "runtime": "native executable"}
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


def benchmark_name(entry: dict[str, Any]) -> str:
    return str(entry.get("benchmark") or entry.get("name") or "unknown")


def stable_case_id(entry: dict[str, Any], profile_version: int) -> str:
    params = dict(entry.get("params") or {})
    if profile_version == 1 and benchmark_name(entry).startswith(f"{V1_SPARSE_PRODUCT_BENCHMARK}."):
        triangle_variant = params.pop("triangleVariant", None)
        if triangle_variant not in {None, V1_TRIANGLE_VARIANT}:
            raise ReportError(
                f"profile v1 cannot normalize developer triangular variant {triangle_variant!r}"
            )
    parts = [f"{key}={params[key]}" for key in sorted(params) if key not in ARM_PARAMETERS]
    suffix = ",".join(parts)
    return f"v{profile_version}:{benchmark_name(entry)}" + (f"[{suffix}]" if suffix else "")


def metric(entry: dict[str, Any]) -> tuple[float, float | None, str, list[Any]]:
    primary = entry.get("primaryMetric") or entry.get("primary_metric") or {}
    score = primary.get("score", entry.get("score"))
    if score is None:
        raise ReportError(f"result row {benchmark_name(entry)} has no score")
    numeric_score = float(score)
    if not math.isfinite(numeric_score):
        raise ReportError(f"result row {benchmark_name(entry)} has a non-finite score")
    error = primary.get("scoreError", primary.get("score_error", entry.get("scoreError")))
    unit = str(primary.get("scoreUnit", primary.get("score_unit", entry.get("unit", "unknown"))))
    raw = primary.get("rawData", primary.get("raw_data", entry.get("rawData", []))) or []
    finite_error = None if error is None or not math.isfinite(float(error)) else float(error)
    return numeric_score, finite_error, unit, raw


def raw_entries(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text())
    entries = document if isinstance(document, list) else document.get("benchmarks", document.get("results", []))
    if not isinstance(entries, list):
        raise ReportError(f"unsupported benchmark JSON in {path.name}")
    return entries


def load_pass(path: Path, arm: str, pass_number: int, profile_version: int) -> list[dict[str, Any]]:
    rows = []
    for entry in raw_entries(path):
        score, error, unit, raw = metric(entry)
        rows.append({
            "case_id": stable_case_id(entry, profile_version),
            "benchmark": benchmark_name(entry),
            "parameters": {str(k): str(v) for k, v in sorted((entry.get("params") or {}).items())},
            "arm": arm,
            "pass": pass_number,
            "score": score,
            "score_error": error,
            "unit": unit,
            "raw_samples": raw,
        })
    return rows


def validate_raw_entry(entry: dict[str, Any], arm: str, settings: dict[str, Any], mode: str) -> None:
    params = entry.get("params") or {}
    selected_arms = {key: str(params[key]) for key in ARM_PARAMETERS if key in params}
    if not selected_arms or set(selected_arms.values()) != {arm}:
        raise ReportError(f"raw row {benchmark_name(entry)} does not select the declared {arm} arm")
    expected = {
        "warmupIterations": settings.get("warmups"),
        "measurementIterations": settings.get("iterations"),
        "warmupTime": f"{settings.get('iteration_time_ms')} ms",
        "measurementTime": f"{settings.get('iteration_time_ms')} ms",
    }
    for field, value in expected.items():
        if entry.get(field) != value:
            raise ReportError(f"raw row {benchmark_name(entry)} has {field}={entry.get(field)!r}; expected {value!r}")
    forks = entry.get("forks", (entry.get("advanced") or {}).get("jvmForks"))
    if forks is not None and int(forks) != settings.get("forks"):
        raise ReportError(f"raw row {benchmark_name(entry)} has forks={forks}; expected {settings.get('forks')}")
    threads = entry.get("threads")
    if threads is not None and int(threads) != settings.get("threads"):
        raise ReportError(f"raw row {benchmark_name(entry)} has threads={threads}; expected {settings.get('threads')}")
    configuration = entry.get("configurationName")
    expected_configuration = "hardwareSmoke" if mode == "smoke" else "hardware"
    if configuration is not None and configuration != expected_configuration:
        raise ReportError(f"raw row {benchmark_name(entry)} uses configuration {configuration}; expected {expected_configuration}")
    if entry.get("mode") != "avgt":
        raise ReportError(f"raw row {benchmark_name(entry)} uses mode {entry.get('mode')}; expected avgt")


def check_pass(rows: list[dict[str, Any]], arm: str, expected: int) -> set[str]:
    ids = [row["case_id"] for row in rows]
    if len(ids) != expected:
        raise ReportError(f"{arm} pass expected {expected} cases but produced {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ReportError(f"{arm} pass contains duplicate stable case IDs")
    return set(ids)


def aggregate_rows(pass_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in pass_rows:
        grouped.setdefault((row["arm"], row["case_id"]), []).append(row)
    result = []
    for (arm, case_id), values in sorted(grouped.items()):
        scores = [value["score"] for value in values]
        reported = [value["score_error"] for value in values if value["score_error"] is not None]
        cross_pass = statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
        uncertainty = max([cross_pass, *reported], default=0.0)
        result.append({
            "schema_version": ROW_SCHEMA,
            "case_id": case_id,
            "benchmark": values[0]["benchmark"],
            "parameters": values[0]["parameters"],
            "arm": arm,
            "score": statistics.fmean(scores),
            "uncertainty": uncertainty,
            "unit": values[0]["unit"],
            "pass_scores": scores,
            "ratio_koblas_to_comparator": None,
            "ratio_uncertainty": None,
        })
    by_case = {(row["arm"], row["case_id"]): row for row in result}
    for row in result:
        if row["arm"] == "built-in":
            continue
        built_in = by_case.get(("built-in", row["case_id"]))
        if built_in is None or built_in["unit"] != row["unit"] or row["score"] == 0:
            continue
        ratio = built_in["score"] / row["score"]
        left = built_in["uncertainty"] / built_in["score"] if built_in["score"] else 0.0
        right = row["uncertainty"] / row["score"]
        row["ratio_koblas_to_comparator"] = ratio
        row["ratio_uncertainty"] = abs(ratio) * math.sqrt(left * left + right * right)
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    (path / "rows.json").write_text(json.dumps({"schema_version": ROW_SCHEMA, "rows": rows}, indent=2, sort_keys=True) + "\n")
    with (path / "rows.csv").open("w", newline="") as handle:
        fields = ["case_id", "benchmark", "parameters", "arm", "score", "uncertainty", "unit", "pass_scores", "ratio_koblas_to_comparator", "ratio_uncertainty"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], sort_keys=True) if key in {"parameters", "pass_scores"} else row[key] for key in fields})


def summary_text(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"koblas hardware report {metadata['run_id']}",
        f"profile: {metadata['workload']['profile']} v{metadata['workload']['version']} ({metadata['mode']})",
        f"target: {metadata['target']}",
        f"commit: {metadata['repository']['commit']} dirty={metadata['repository']['dirty']}",
        f"host: {metadata['host']['os']} {metadata['host']['architecture']} / {metadata['host']['cpu_model']}",
        f"quality: {metadata['quality']['annotation']}",
        "",
    ]
    for arm in metadata["arms"]:
        arm_rows = [row for row in rows if row["arm"] == arm]
        ratios = [row["ratio_koblas_to_comparator"] for row in arm_rows if row["ratio_koblas_to_comparator"] is not None]
        text = f"{arm}: {len(arm_rows)} cases"
        if ratios:
            text += f", median koblas/comparator time ratio {statistics.median(ratios):.3f}"
        lines.append(text)
    lines.extend(["", "Ratios exist only for matching case IDs, profile versions, units, and settings."])
    return "\n".join(lines) + "\n"


def checksums(directory: Path) -> None:
    lines = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(directory).as_posix()}")
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def safe_members(archive: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    base = destination.resolve()
    for member in members:
        resolved = (destination / member.name).resolve()
        if base != resolved and base not in resolved.parents:
            raise ReportError("bundle contains an unsafe archive path")
    return members


def bundle_directory(path: Path):
    class BundleContext:
        temporary: tempfile.TemporaryDirectory[str] | None = None

        def __enter__(self) -> Path:
            if path.is_dir():
                return path
            self.temporary = tempfile.TemporaryDirectory(prefix="koblas-bundle-")
            destination = Path(self.temporary.name)
            with tarfile.open(path, "r:gz") as archive:
                archive.extractall(destination, members=safe_members(archive, destination), filter="data")
            roots = [item for item in destination.iterdir() if item.is_dir()]
            if len(roots) != 1:
                raise ReportError("bundle archive must contain one root directory")
            return roots[0]

        def __exit__(self, *_: Any) -> None:
            if self.temporary is not None:
                self.temporary.cleanup()

    return BundleContext()


def validate_directory(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = ["metadata.json", "rows.json", "rows.csv", "summary.txt", "SHA256SUMS", "raw"]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise ReportError(f"bundle is missing: {', '.join(missing)}")
    manifest: dict[str, str] = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ReportError("invalid checksum manifest") from error
        relative_path = Path(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ReportError(f"invalid checksum entry: {relative}")
        if relative in manifest:
            raise ReportError(f"duplicate checksum entry: {relative}")
        manifest[relative] = digest
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ReportError(f"checksum mismatch: {relative}")
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(manifest) != actual_files:
        omitted = sorted(actual_files - set(manifest))
        unexpected = sorted(set(manifest) - actual_files)
        details = ", ".join([*(f"unlisted {name}" for name in omitted), *(f"missing {name}" for name in unexpected)])
        raise ReportError(f"checksum manifest does not cover the bundle: {details}")
    metadata = json.loads((directory / "metadata.json").read_text())
    row_document = json.loads((directory / "rows.json").read_text())
    if metadata.get("schema_version") != BUNDLE_SCHEMA or row_document.get("schema_version") != ROW_SCHEMA:
        raise ReportError("unsupported bundle or row schema version")
    rows = row_document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ReportError("bundle has no machine-readable rows")
    arms = metadata.get("arms")
    expected = metadata.get("expected_cases")
    workload = metadata.get("workload", {})
    profile_version = workload.get("version")
    settings = workload.get("settings")
    mode = metadata.get("mode")
    required_settings = {"warmups", "iterations", "iteration_time_ms", "forks", "threads"}
    if (
        not isinstance(arms, list)
        or not arms
        or not isinstance(expected, dict)
        or not isinstance(profile_version, int)
        or not isinstance(settings, dict)
        or not required_settings.issubset(settings)
        or mode not in {"smoke", "standard"}
    ):
        raise ReportError("bundle metadata is incomplete")
    row_arms = {row.get("arm") for row in rows}
    if row_arms != set(arms):
        raise ReportError("machine-readable row arms do not match metadata")
    all_pass_rows: list[dict[str, Any]] = []
    for arm in arms:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        arm_expected = expected.get(arm)
        if not isinstance(arm_expected, int) or arm_expected < 1 or len(arm_rows) != arm_expected:
            raise ReportError(f"bundle has {len(arm_rows)} {arm} rows; expected {arm_expected}")
        aggregate_ids = [row.get("case_id") for row in arm_rows]
        if len(set(aggregate_ids)) != len(aggregate_ids):
            raise ReportError(f"bundle has duplicate aggregate case IDs for {arm}")
        pass_ids: list[set[str]] = []
        for pass_number in (1, 2):
            raw = directory / "raw" / arm / f"pass-{pass_number}.json"
            if not raw.is_file():
                raise ReportError(f"bundle is missing raw {arm} pass {pass_number}")
            log = directory / "raw" / arm / f"pass-{pass_number}.log"
            if not log.is_file():
                raise ReportError(f"bundle is missing raw {arm} pass {pass_number} log")
            if FAILURE_PATTERN.search(log.read_text(errors="replace")):
                raise ReportError(f"bundle contains a benchmark fork failure for {arm} pass {pass_number}")
            entries = raw_entries(raw)
            for entry in entries:
                validate_raw_entry(entry, arm, settings, mode)
            raw_rows = load_pass(raw, arm, pass_number, profile_version)
            pass_ids.append(check_pass(raw_rows, arm, arm_expected))
            all_pass_rows.extend(raw_rows)
        if pass_ids[0] != pass_ids[1]:
            raise ReportError(f"{arm} raw passes do not contain identical case IDs")
        if pass_ids[0] != set(aggregate_ids):
            raise ReportError(f"{arm} aggregate rows do not match raw case IDs")
        for row in arm_rows:
            if row.get("schema_version") != ROW_SCHEMA:
                raise ReportError(f"unsupported row schema version for {row.get('case_id')}")
            scores = row.get("pass_scores")
            if not isinstance(scores, list) or len(scores) != 2 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scores):
                raise ReportError(f"aggregate row {row.get('case_id')} does not contain two finite pass scores")
            for field in ("score", "uncertainty"):
                value = row.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ReportError(f"aggregate row {row.get('case_id')} has invalid {field}")
    recomputed = aggregate_rows(all_pass_rows)
    reported_by_key = {(row["arm"], row["case_id"]): row for row in rows}
    recomputed_by_key = {(row["arm"], row["case_id"]): row for row in recomputed}
    if reported_by_key.keys() != recomputed_by_key.keys():
        raise ReportError("aggregate row keys do not match raw measurements")
    for key, reported in reported_by_key.items():
        if reported != recomputed_by_key[key]:
            raise ReportError(f"aggregate row {key[1]} for {key[0]} does not match raw measurements")
    return metadata, rows


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
        settings = catalog["settings"] if not smoke else {**catalog["settings"], "warmups": 1, "iterations": 1, "iteration_time_ms": 20}
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
            "repository": {"commit": output(["git", "rev-parse", "HEAD"]), "dirty": output(["git", "status", "--porcelain"]) not in {"", "unknown"}},
            "workload": {"profile": catalog["profile"], "version": catalog["profile_version"], "seed": catalog["seed"], "settings": settings},
            "host": preflight["host"],
            "runtime": {**preflight["runtime"], "measured_process_lines": sorted(set(runtime_lines)) or ["unknown"]},
            "implementations": sorted(set(resolved_lines)) or ["unknown"],
            "comparators": preflight["comparators"],
            "threading": {"benchmark_threads": 1, "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            "affinity": f"taskset -c {args.cores}" if args.cores else "none",
            "tuning_overrides": tuning,
            "quality": {"concurrent_runs_detected": sorted(concurrent_seen), "annotation": "concurrent report activity detected; retain uncertainty and raw passes" if concurrent_seen else "no concurrent koblas report runner detected; other machine activity was not excluded"},
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
