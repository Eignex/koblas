"""Pure processing and validation for koblas hardware benchmark bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


BUNDLE_SCHEMA = 1
ROW_SCHEMA = 1
FAILURE_PATTERN = re.compile(r"<failure>|EXCEPTION: <ERROR>|Benchmark fork reported", re.IGNORECASE)
ARM_PARAMETERS = {"denseArm", "sparseArm", "comparator"}
V1_SPARSE_PRODUCT_BENCHMARK = "com.eignex.koblas.bench.SparseProductHostBenchmark"
V1_TRIANGLE_VARIANT = "upper-nontrans-nonunit"


class ReportError(RuntimeError):
    pass


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


@contextmanager
def bundle_directory(path: Path):
    if path.is_dir():
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="koblas-bundle-") as temporary:
        destination = Path(temporary)
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(destination, members=safe_members(archive, destination), filter="data")
        roots = [item for item in destination.iterdir() if item.is_dir()]
        if len(roots) != 1:
            raise ReportError("bundle archive must contain one root directory")
        yield roots[0]


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


def validate_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with bundle_directory(path) as directory:
        return validate_directory(directory)
