#!/usr/bin/env python3
"""Fit conservative engine preferences from repeated, paired capture-report outputs."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from pathlib import Path

MAX = 2**31 - 1
VECTOR = dict(dot="Dot", sum="Sum", nrm2="Nrm2", iamax="Iamax", asum="Asum",
              axpy="Axpy", scal="Scale", swap="Swap", rot="Rot")
MATRIX = dict(gemv="Gemv", symv="Symv", ger="Ger", syr="Syr", syr2="Syr2", trmv="Trmv",
              trsv="Trsv", gemm="Gemm", gemmt="Gemmt", symm="Symm", syrk="Syrk",
              syr2k="Syr2k", trmm="Trmm", trsm="Trsm")


def measurements(directory, baseline, candidate, backend=None):
    passes = sorted(directory.glob("pass-*"))
    passes = [p for p in passes if p.is_dir()]
    if len(passes) < 3:
        raise ValueError("at least three paired capture rounds are required")
    paired, provenance = {}, {}
    for path in passes:
        metadata = (path / "metadata.txt").read_text()
        fields = dict(line.split("=", 1) for line in metadata.splitlines() if "=" in line)
        if fields.get("status") != "complete" or int(fields.get("requested_samples", 0)) < 3 or int(fields.get("target_ns", 0)) < 10_000_000:
            raise ValueError(f"{path}: incomplete capture or smoke timing settings")
        exported = re.findall(r"/caps\[" + re.escape(backend or "") + r":([^]]*)\]", metadata)
        supported = set(exported[-1].split(',')) if exported else None
        if backend and supported is None and (backend == "jvm-simd" or candidate.endswith("-default")):
            raise ValueError("Combined fitting needs capability metadata; recapture with the current harness")
        rows = []
        for target in (baseline, candidate):
            with (path / f"{target}.csv").open() as source:
                rows.append({row["case"]: row for row in csv.DictReader(source)})
        for case in rows[0].keys() & rows[1].keys():
            left, right = rows[0][case], rows[1][case]
            if supported is not None and (key := shape(case)) and key[0] not in supported:
                continue
            if left["status"] != "ok" or right["status"] != "ok":
                continue
            if left["timing_mode"] != right["timing_mode"]:
                raise ValueError(f"{case}: different timing boundaries")
            times = [float(row["median_ns"]) for row in (left, right)]
            if not all(math.isfinite(t) and t > 0 for t in times):
                raise ValueError(f"{case}: invalid timings")
            paired.setdefault(case, []).append(times[1] / times[0])
        for name in ("metadata.txt", f"{baseline}.csv", f"{candidate}.csv"):
            provenance[str(path / name)] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    return {case: ratios for case, ratios in paired.items() if len(ratios) == len(passes)}, provenance


def shape(case):
    operation, dimensions, _, *options = case.split("+")
    flags = dict(option.split("=", 1) for option in options)
    dims = [int(n) for n in dimensions.split("x")]
    if operation in VECTOR:
        return VECTOR[operation], dims[0], 1, 1, False
    if operation not in MATRIX:
        return None
    borrowed = flags.get("layout") == "borrowed"
    name = MATRIX[operation]
    if operation in ("gemv", "trmv", "trsv") and flags.get("transA") == "T":
        name += "Transposed"
    if operation == "gemm":
        return name, *dims, borrowed
    if len(dims) == 1:
        return name, dims[0], dims[0], 1, borrowed
    if operation in ("gemmt", "syrk", "syr2k"):
        return name, dims[0], dims[0], dims[1], borrowed
    if operation in ("symm", "trmm", "trsm"):
        rows, columns = dims
        order = columns if flags.get("side") == "R" else rows
        return name, rows, columns, order, borrowed
    return name, *dims, 1, borrowed


def fit(paired, backend):
    groups = {}
    for case, ratios in paired.items():
        key = shape(case)
        if key:
            groups.setdefault(key, []).extend(ratios)
    wins = {key: statistics.median(ratios) <= 0.95 and max(ratios) <= 1.02
            for key, ratios in groups.items()}
    rules = []
    for operation in sorted({key[0] for key in groups}):
        keys = sorted(key for key in groups if key[0] == operation)
        if operation in VECTOR.values():
            # A threshold must keep winning at every larger measured length, including all variants.
            start = next((i for i in range(len(keys)) if all(wins[k] for k in keys[i:])), None)
            if start is not None:
                rules.append([backend, operation, keys[start][1], keys[-1][1], 0, MAX, 0, MAX, 0, "true", "*"])
        else:
            # Matrix shapes are bounded to observed extents; no square-only extrapolation to skinny calls.
            for key in keys:
                if wins[key]:
                    _, rows, columns, depth, borrowed = key
                    rules.append([backend, operation, rows, rows, columns, columns, depth, depth, 0, "*", str(borrowed).lower()])
    if backend == "jvm-simd":
        rules.append([backend, "Panels,Products,Triangular", 0, MAX, 0, MAX, 0, MAX, 0, "*", "*"])
    return rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("native", "jvm"), required=True)
    parser.add_argument("--backend", choices=("onemkl", "openblas", "aocl", "armpl", "accelerate", "jvm-simd"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", help="capture target for an explicitly configured combined engine")
    parser.add_argument("--baseline", choices=("native", "jvm-scalar", "jvm-simd"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--cases", type=Path, help="custom training cases; defaults to a regular dense grid")
    parser.add_argument("--validation-cases", default="koblas-bench/cases-addressed.txt")
    parser.add_argument("--native-executable", type=Path)
    parser.add_argument("--os", default="Linux")
    parser.add_argument("--arch", default="X86_64")
    parser.add_argument("--cpu", default="Intel")
    args = parser.parse_args()
    if args.capture == bool(args.results):
        parser.error("choose --capture or --results")
    if args.backend == "jvm-simd" and args.runtime != "jvm":
        parser.error("jvm-simd requires --runtime jvm")
    baseline = args.baseline or ("native" if args.runtime == "native" else
                                 "jvm-scalar" if args.backend == "jvm-simd" else "jvm-simd")
    if baseline.startswith("jvm") != (args.runtime == "jvm"):
        parser.error("baseline must use the selected runtime")
    candidate = args.candidate or args.backend + ("-jvm" if args.runtime == "jvm" and args.backend != "jvm-simd" else "")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.cases is None:
        args.cases = args.output / "training-cases.txt"
        cases = [f"{op}+{n}+uniform" for op in VECTOR for n in (32, 64, 96, 128, 256, 1024, 4096)]
        for n in (8, 16, 32, 64, 128):
            cases += [f"gemm+{n}x{n}x{n}+uniform", f"gemv+{n}x{n}+uniform",
                      f"syrk+{n}x{n}+uniform+uplo=L+transA=N",
                      f"trsm+{n}x{n}+triangular+side=L+uplo=L+transA=N+diag=N"]
        cases += ["gemm+16x16x1024+uniform", "gemm+16x128x64+uniform", "gemm+128x16x64+uniform"]
        args.cases.write_text("\n".join(cases) + "\n")
    results = args.results or args.output / "capture"
    header = ["koblas-profile-v1", f"{args.runtime}-{args.backend}-recalculated-v1", args.runtime,
              args.os, args.arch, args.cpu, str(results), "see evidence.json for paired measurements and hashes"]
    def write_profile(path, preferences):
        path.write_text("\n".join("\t".join(map(str, row)) for row in [header, *preferences]) + "\n")
    capture_env = dict(os.environ)
    if args.capture and args.backend == "jvm-simd" and args.candidate is None:
        candidate = "jvm-default"
        forcing = [[args.backend, ",".join(VECTOR.values()), 0, MAX, 0, MAX, 0, MAX, 0, "true", "*"]]
        force_path = args.output / "force.tsv"
        write_profile(force_path, forcing + fit({}, args.backend))
        capture_env["KOBLAS_BENCH_ENGINE_PROFILE"] = str(force_path.resolve())
    if args.capture:
        command = ["bash", "koblas-bench/capture-report.sh", "--targets", f"{baseline},{candidate}",
                   "--cases", str(args.cases), "--rounds", "3", "--samples", "5", "--warmups", "5",
                   "--target-ms", "100", "--forks", "2", "--output", str(results)]
        if args.backend != "jvm-simd":
            command += ["--libraries", args.backend]
        if args.native_executable:
            command += ["--native-executable", str(args.native_executable)]
        subprocess.run(command, check=True, env=capture_env)
    paired, provenance = measurements(results, baseline, candidate, args.backend)
    if not paired:
        raise ValueError("no complete paired measurements")
    rules = fit(paired, args.backend)
    write_profile(args.output / "profile.tsv", rules)
    (args.output / "evidence.json").write_text(json.dumps(dict(
        baseline=baseline, candidate=candidate, source_hashes=provenance,
        paired_ratios=paired, initial_rules=rules, criterion="median <= 0.95 and every round <= 1.02",
    ), indent=2) + "\n")
    if args.capture:
        validation_cases = args.output / "validation-cases.txt"
        cases = Path(args.cases).read_text().splitlines() + Path(args.validation_cases).read_text().splitlines()
        validation_cases.write_text("\n".join(dict.fromkeys(cases)) + "\n")
        command[command.index("--targets") + 1] = f"{baseline},{args.runtime}-default"
        command[command.index("--cases") + 1] = str(validation_cases)
        for attempt in range(1, 4):
            validation = args.output / f"validation-{attempt}"
            command[command.index("--output") + 1] = str(validation)
            subprocess.run(command, check=True, env=dict(os.environ,
                           KOBLAS_BENCH_ENGINE_PROFILE=str((args.output / "profile.tsv").resolve())))
            ratios, hashes = measurements(validation, baseline, f"{args.runtime}-default")
            regressions = {case: values for case, values in ratios.items()
                           if statistics.median(values) > 1.05 and min(values) > 1.02}
            covered = {case: values for case, values in ratios.items() if (key := shape(case)) and any(
                key[0] == rule[1] and (rule[10] == "*" or (rule[10] == "true") == key[4]) and
                all(rule[low] <= n <= rule[low + 1] for low, n in zip((2, 4, 6), key[1:4])) for rule in rules)}
            selected = {case: values for case, values in regressions.items() if case in covered}
            (args.output / f"validation-{attempt}.json").write_text(json.dumps(dict(
                paired_ratios=ratios, source_hashes=hashes, rules=rules, regressions=selected,
                control_regressions={case: values for case, values in regressions.items() if case not in covered}),
                indent=2) + "\n")
            if ratios and not selected:
                break
            refined = fit(covered, args.backend)
            if attempt == 3 or refined == rules:
                raise ValueError("combined profile validation failed; inspect retained validation reports")
            rules = refined
            write_profile(args.output / "profile.tsv", rules)
    print(f"Wrote {len(rules)} rules to {args.output / 'profile.tsv'}; retained evidence is available beside the profile.")


if __name__ == "__main__":
    main()
