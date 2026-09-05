#!/usr/bin/env python3
"""Validate reviewed benchmark coverage against the public numerical operation surface, optionally against
benchmark JSON reports."""
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
PUBLIC_FUNCTION = re.compile(r"\bpublic\s+(?:[A-Za-z]+\s+)*fun\s+")
TYPE_DECLARATION = re.compile(r"\b(?:class|interface|object)\s+(\w+)")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT = re.compile(r"//[^\n]*")
FUNCTION_TYPE_PARAMETER_NAME = re.compile(r"(?<=[(,])\s*\w+\s*:\s*")
HEADER_END = re.compile(r"\b(?:fun|class|interface|object|typealias|val|var)\b")

# These are the public operation facades. Storage construction, backend configuration, and lifecycle methods are
# intentionally not numerical operations. Keep this list alongside a new facade so the inventory remains complete.
PUBLIC_NUMERICAL_SOURCES = (
    "F64Givens.kt",
    "F64ModifiedGivens.kt",
    "MatrixNorms.kt",
    "MatrixOps.kt",
    "MatrixScaling.kt",
    "MatrixSlices.kt",
    "Operators.kt",
    "VectorOps.kt",
    "dense/Cholesky.kt",
    "dense/F64Blas.kt",
    "dense/F64Decompositions.kt",
    "dense/F64Kernels.kt",
    "dense/F64LinearAlgebra.kt",
    "dense/F64LuDecomposition.kt",
    "dense/FactorSnapshots.kt",
    "dense/Triangular.kt",
    "sparse/F64SparseBlas.kt",
    "sparse/F64SparseDecompositions.kt",
    "sparse/F64SparseFactorization.kt",
    "sparse/F64SparseKernels.kt",
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


def without_comments(text):
    """KDoc and ordinary comments quote declarations, so they are removed before anything is scanned."""
    return LINE_COMMENT.sub("", BLOCK_COMMENT.sub("", text))


def matching(text, index, opening, closing):
    depth = 0
    while index < len(text):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    fail(f"unbalanced {opening}{closing} in {text[:40]!r}")


def top_level_split(text, separator):
    parts, depth, current = [], 0, []
    for character in text:
        if character in "(<[":
            depth += 1
        elif character in ")>]":
            depth -= 1
        if character == separator and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def parameter_type(parameter):
    """One parameter reduced to its declared type: no name, no default, no names inside a function type."""
    declared = top_level_split(parameter, ":")
    if len(declared) < 2:
        fail(f"parameter without a declared type: {parameter}")
    without_default = top_level_split(declared[1], "=")[0]
    return FUNCTION_TYPE_PARAMETER_NAME.sub("", without_default).replace(" ", "").replace("\n", "")


def type_bodies(text):
    """Brace ranges of the named classes, interfaces, and objects, so a member can be attributed to its type."""
    bodies = []
    for match in TYPE_DECLARATION.finditer(text):
        index, depth = match.end(), 0
        while index < len(text):
            character = text[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            elif depth == 0 and character == "{":
                bodies.append((index, matching(text, index, "{", "}"), match.group(1)))
                break
            elif depth == 0 and HEADER_END.match(text, index):
                # A declaration with no body of its own; the next brace opens something else entirely.
                break
            index += 1
    return bodies


def declared_operations(text):
    """Every public function as `Qualifier.name(ParameterTypes)`, which keeps overloads distinct."""
    text = without_comments(text)
    bodies = type_bodies(text)
    operations = []
    for match in PUBLIC_FUNCTION.finditer(text):
        start = match.end()
        if text[start] == "<":
            start = matching(text, start, "<", ">") + 1
        opening = text.find("(", start)
        head = " ".join(text[start:opening].split())
        closing = matching(text, opening, "(", ")")
        types = [parameter_type(parameter) for parameter in top_level_split(text[opening + 1 : closing], ",")]
        receiver, _, name = head.rpartition(".")
        enclosing = [name for start, end, name in bodies if start < match.start() < end]
        qualifier = receiver or (enclosing[-1] if enclosing else "")
        operations.append(f"{qualifier}.{name}({','.join(types)})" if qualifier else f"{name}({','.join(types)})")
    return operations


def public_numerical_operations(source_root=NUMERICAL_SOURCE):
    found = set()
    for relative in PUBLIC_NUMERICAL_SOURCES:
        source = source_root / relative
        if not source.is_file():
            fail(f"public numerical API source is missing: {source}")
        for operation in declared_operations(source.read_text(encoding="utf-8")):
            key = f"{relative}:{operation}"
            if key in found:
                fail(f"two public declarations share the operation key {key}")
            found.add(key)
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
