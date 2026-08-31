#!/usr/bin/env python3
"""Validate reviewed benchmark coverage, optionally against benchmark JSON reports."""
import csv
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "commonMain"
MODIFIER = r"(?:public|internal|private|protected|open|override|suspend|inline)"
BENCHMARK = re.compile(rf"@Benchmark(?:\([^)]*\))?\s*(?:{MODIFIER}\s+)*fun\s+(\w+)")
CLASS = re.compile(r"class\s+(\w+Benchmark)\b")

def fail(message):
    raise SystemExit(f"benchmark coverage: {message}")

def manifest(path):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = ["api_signature", "benchmark_id", "status", "notes"]
        if reader.fieldnames != required:
            fail("manifest header must be " + "\t".join(required))
        rows = list(reader)
    if not rows:
        fail("manifest has no data rows")
    signatures, listed, benchmarked = set(), set(), set()
    for number, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            fail(f"malformed row {number}")
        signature, identifier, status, notes = (row[key].strip() for key in required)
        if not signature or not identifier or status not in {"benchmarked", "excluded"}:
            fail(f"malformed row {number}")
        if signature in signatures:
            fail(f"duplicate API signature: {signature}")
        if status == "excluded" and not notes:
            fail(f"excluded row {number} needs notes")
        signatures.add(signature)
        listed.add(identifier)
        if status == "benchmarked":
            benchmarked.add(identifier)
    return listed, benchmarked

def source_benchmarks():
    found = set()
    for source in SOURCE.rglob("*.kt"):
        text = source.read_text(encoding="utf-8")
        class_matches = list(CLASS.finditer(text))
        if not class_matches:
            continue
        bounds = [match.start() for match in class_matches] + [len(text)]
        for index, class_match in enumerate(class_matches):
            body = text[bounds[index] : bounds[index + 1]]
            for method in BENCHMARK.findall(body):
                found.add(f"com.eignex.koblas.bench.{class_match.group(1)}.{method}")
    return found

def report_benchmarks(paths):
    found = set()
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            fail(f"cannot read JSON report {path}: {error}")
        for row in document:
            identifier = row.get("benchmark")
            if identifier:
                found.add(identifier)
    return found

def main(arguments):
    if not arguments:
        fail("usage: check-benchmark-coverage.py MANIFEST [REPORT.json ...]")
    listed, benchmarked = manifest(pathlib.Path(arguments[0]))
    discovered = source_benchmarks()
    missing_methods = listed - discovered
    unlisted_methods = discovered - listed
    if missing_methods:
        fail("manifest names nonexistent benchmark methods: " + ", ".join(sorted(missing_methods)))
    if unlisted_methods:
        fail("benchmark methods absent from manifest: " + ", ".join(sorted(unlisted_methods)))
    if len(arguments) > 1:
        missing = benchmarked - report_benchmarks([pathlib.Path(path) for path in arguments[1:]])
        if missing:
            fail("report is missing benchmark methods: " + ", ".join(sorted(missing)))
    print(f"benchmark coverage OK: {len(benchmarked)} benchmarked methods")

if __name__ == "__main__":
    main(sys.argv[1:])

