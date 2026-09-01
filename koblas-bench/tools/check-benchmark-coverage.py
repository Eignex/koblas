#!/usr/bin/env python3
"""Validate reviewed benchmark coverage against the public numerical operation surface."""
import csv
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCHMARK_SOURCE = ROOT / "src" / "commonMain"
NUMERICAL_SOURCE = ROOT.parent / "koblas" / "src" / "commonMain" / "kotlin" / "com" / "eignex" / "koblas"
INVENTORY = ROOT / "public-numerical-api.tsv"
MODIFIER = r"(?:public|internal|private|protected|open|override|suspend|inline)"
BENCHMARK = re.compile(rf"@Benchmark(?:\([^)]*\))?\s*(?:{MODIFIER}\s+)*fun\s+(\w+)")
CLASS = re.compile(r"class\s+(\w+Benchmark)\b")
PUBLIC_FUNCTION = re.compile(r"public\s+(?:[A-Za-z]+\s+)*fun\s+(?:[A-Za-z0-9_<>?.]+\s*\.)?(\w+)\s*\(")

# These are the public operation facades. Storage construction, backend configuration, and lifecycle methods are
# intentionally not numerical operations. Keep this list alongside a new facade so the inventory remains complete.
PUBLIC_NUMERICAL_SOURCES = (
    "F64Givens.kt",
    "MatrixOps.kt",
    "Operators.kt",
    "VectorOps.kt",
    "dense/Cholesky.kt",
    "dense/F64Blas.kt",
    "dense/F64Decompositions.kt",
    "dense/F64LinearAlgebra.kt",
    "dense/F64LuDecomposition.kt",
    "dense/Triangular.kt",
    "sparse/F64SparseBlas.kt",
    "sparse/F64SparseDecompositions.kt",
    "sparse/F64SparseFactorization.kt",
    "sparse/F64SparseQrFactorization.kt",
    "sparse/SparseOps.kt",
    "sparse/basis/F64BasisSolver.kt",
    "sparse/basis/F64BasisSolvers.kt",
)

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
        if not signature or status not in {"benchmarked", "excluded"}:
            fail(f"malformed row {number}")
        if signature in signatures:
            fail(f"duplicate API signature: {signature}")
        if status == "benchmarked" and not identifier:
            fail(f"benchmarked row {number} needs a benchmark method")
        if status == "excluded" and (identifier or not notes):
            fail(f"excluded row {number} needs notes and no benchmark method")
        signatures.add(signature)
        if status == "benchmarked":
            listed.add(identifier)
            benchmarked.add(identifier)
    return signatures, listed, benchmarked


def public_numerical_operations(source_root=NUMERICAL_SOURCE):
    found = set()
    for relative in PUBLIC_NUMERICAL_SOURCES:
        source = source_root / relative
        if not source.is_file():
            fail(f"public numerical API source is missing: {source}")
        for method in PUBLIC_FUNCTION.findall(source.read_text(encoding="utf-8")):
            found.add(f"{relative}:{method}")
    return found


def api_inventory(path, signatures, operations):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = ["api_surface", "api_signature"]
        if reader.fieldnames != required:
            fail("API inventory header must be " + "\t".join(required))
        rows = list(reader)
    if not rows:
        fail("API inventory has no data rows")
    listed = set()
    for number, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            fail(f"malformed API inventory row {number}")
        surface, signature = (row[key].strip() for key in required)
        if not surface or not signature:
            fail(f"malformed API inventory row {number}")
        if surface in listed:
            fail(f"duplicate public numerical API surface: {surface}")
        if signature not in signatures:
            fail(f"API inventory row {number} names unlisted API signature: {signature}")
        listed.add(surface)
    missing = operations - listed
    stale = listed - operations
    if missing:
        fail("public numerical operations absent from inventory: " + ", ".join(sorted(missing)))
    if stale:
        fail("API inventory names nonexistent public numerical operations: " + ", ".join(sorted(stale)))

def source_benchmarks(source_root=BENCHMARK_SOURCE):
    found = set()
    for source in source_root.rglob("*.kt"):
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
    signatures, listed, benchmarked = manifest(pathlib.Path(arguments[0]))
    api_inventory(INVENTORY, signatures, public_numerical_operations())
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
