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
import sys
from pathlib import Path

MAX = 2**31 - 1


def measurements(directory, baseline, candidate, backend=None, strict=False):
    passes = sorted(directory.glob("pass-*"))
    passes = [p for p in passes if p.is_dir()]
    if len(passes) < 3:
        raise ValueError("at least three paired capture rounds are required")
    paired, provenance, shapes, providers = {}, {}, {}, {}
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
        if strict and rows[0].keys() != rows[1].keys():
            raise ValueError("validation arms contain different cases")
        for case in rows[0].keys() & rows[1].keys():
            left, right = rows[0][case], rows[1][case]
            if left["status"] != "ok" or right["status"] != "ok":
                if strict:
                    raise ValueError(f"{case}: validation contains an untimed case")
                continue
            if left["timing_mode"] != right["timing_mode"]:
                raise ValueError(f"{case}: different timing boundaries")
            if left["routing_key"] != right["routing_key"]:
                raise ValueError(f"{case}: different routing dimensions")
            fields = left["routing_key"].split(":")
            key = (fields[0], *map(int, fields[1:4]), fields[4] == "true", int(fields[5]), fields[6] == "true") if fields[0] else None
            provider = right["provider"]
            if not provider or providers.setdefault(case, provider) != provider:
                raise ValueError(f"{case}: missing or inconsistent selected provider")
            if case in shapes and shapes[case] != key:
                raise ValueError(f"{case}: routing dimensions changed between rounds")
            shapes[case] = key
            if backend and provider != backend:
                continue
            if supported is not None and key and key[0] not in supported:
                continue
            times = [float(row["median_ns"]) for row in (left, right)]
            if not all(math.isfinite(t) and t > 0 for t in times):
                raise ValueError(f"{case}: invalid timings")
            paired.setdefault(case, []).append(times[1] / times[0])
        for name in ("metadata.txt", f"{baseline}.csv", f"{candidate}.csv"):
            provenance[str(path / name)] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    complete = {case: ratios for case, ratios in paired.items() if len(ratios) == len(passes)}
    if strict and (not complete or len(complete) != len(rows[0])):
        raise ValueError("validation requires complete paired timings for every case")
    return complete, provenance, shapes, providers


def measured_cpu(directory, requested=None):
    models = set()
    for path in directory.glob("pass-*/metadata.txt"):
        detected = [line.partition("=")[2] for line in path.read_text().splitlines()
                    if line.startswith("profile_cpu_model=")]
        if not detected or not all(detected):
            raise ValueError(f"{path}: missing runtime CPU identity; recapture with the current harness")
        models.update(detected)
    if len(models) != 1:
        raise ValueError("profile fitting requires one runtime CPU model; use separate captures per model")
    model = models.pop()
    if requested == "*":
        return requested
    if model == "*" or requested is not None and requested != model:
        raise ValueError("CPU model is not detectable by the runtime; use --cpu '*' for a platform wide profile")
    return model


def fit(paired, backend, shapes):
    groups = {}
    for case, ratios in paired.items():
        key = shapes[case]
        if key:
            groups.setdefault(key, []).extend(ratios)
    wins = {key: statistics.median(ratios) <= 0.95 and max(ratios) <= 1.02
            for key, ratios in groups.items()}
    rules = []
    for operation, contiguous in sorted({(key[0], key[6]) for key in groups}):
        keys = sorted(key for key in groups if (key[0], key[6]) == (operation, contiguous))
        if keys[0][5] == 1:
            # A threshold must keep winning at every larger measured length, including all variants.
            start = next((i for i in range(len(keys)) if all(wins[k] for k in keys[i:])), None)
            if start is not None:
                rules.append([backend, operation, keys[start][1], keys[-1][1], 0, MAX, 0, MAX, 0, str(contiguous).lower(), "*"])
        else:
            # Matrix shapes are bounded to observed extents; no square-only extrapolation to skinny calls.
            for key in keys:
                if wins[key]:
                    _, rows, columns, depth, borrowed, _, _ = key
                    rules.append([backend, operation, rows, rows, columns, columns, depth, depth, 0, "*", str(borrowed).lower()])
    if backend == "jvm-simd":
        rules.append([backend, "Panels,Products,Triangular", 0, MAX, 0, MAX, 0, MAX, 0, "*", "*"])
    return rules


def training_cases():
    vectors = ("dot", "copy", "sum", "nrm2", "iamax", "asum", "axpy", "scal", "swap", "rot")
    cases = [f"{op}+{n}+uniform{stride}" for op in vectors
             for n in (2, 4, 8, 16, 32, 64, 96, 128, 256, 1024, 65536, 1_000_000)
             for stride in ("", "+stride=2", "+stride=-1", "+stride=-2")]
    for n in (8, 16, 32, 64, 128, 256, 512):
        cases += [f"{op}+{n}+uniform+uplo=L" for op in ("symv", "syr", "syr2")]
        cases += [f"{op}+{n}+triangular+uplo=L+transA={trans}+diag=N"
                  for op in ("trmv", "trsv") for trans in ("N", "T")]
        cases += [f"{op}+{n}x{n}+uniform{transpose}" for op in ("gemv", "ger")
                  for transpose in (("", "+transA=T") if op == "gemv" else ("",))]
        cases += [f"gemm+{n}x{n}x{n}+uniform", f"symm+{n}x{n}+uniform+side=L+uplo=L",
                  f"gemmt+{n}x{n}+uniform+uplo=L+transA=N+transB=N"]
        cases += [f"{op}+{n}x{n}+uniform+uplo=L+transA={trans}"
                  for op in ("syrk", "syr2k") for trans in ("N", "T")]
        cases += [f"{op}+{n}x{n}+triangular+side=L+uplo=L+transA=N+diag=N" for op in ("trsm", "trmm")]
    for m, n, k in ((16, 128, 64), (128, 16, 64), (31, 65, 129),
                    *((16, 16, k) for k in (256, 512, 1024, 2048, 8192))):
        cases += [f"gemm+{m}x{n}x{k}+uniform{layout}" for layout in ("", "+layout=borrowed")]
    cases += [f"{op}+{n * 4}+sparse-uniform+density=0.25" for n in (2, 4, 8, 16, 32, 64, 256, 1024)
              for op in ("spdot", "spdot-sparse", "spaxpy", "spscatter", "spgather", "spgather-zero", "spnrm2-indexed")]
    return cases


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
    parser.add_argument("--cpu", help="explicit target model; defaults to the runtime CPU identity; use * explicitly for platform wide profiles")
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
        cases = training_cases()
        args.cases.write_text("\n".join(cases) + "\n")
    results = args.results or args.output / "capture"
    header = ["koblas-profile-v1", f"{args.runtime}-{args.backend}-recalculated-v1", args.runtime,
              args.os, args.arch, args.cpu or "*", str(results), "see evidence.json for paired measurements and hashes"]
    def write_profile(path, preferences):
        path.write_text("\n".join("\t".join(map(str, row)) for row in [header, *preferences]) + "\n")
    capture_env = dict(os.environ)
    if args.capture and args.backend == "jvm-simd" and args.candidate is None:
        candidate = "jvm-default"
        forcing = [[args.backend, "Dot,Sum,Nrm2,Iamax,Asum,Axpy,IndexedDot", 0, MAX, 0, MAX, 0, MAX, 0, "true", "*"]]
        force_path = args.output / "force.tsv"
        write_profile(force_path, forcing + fit({}, args.backend, {}))
        capture_env["KOBLAS_BENCH_ENGINE_PROFILE"] = str(force_path.resolve())
    if args.capture:
        command = ["bash", "koblas-bench/capture-report.sh", "--targets", f"{baseline},{candidate}",
                   "--cases", str(args.cases), "--rounds", "3", "--samples", "5", "--warmups", "20",
                   "--target-ms", "200", "--forks", "2", "--output", str(results)]
        if sys.platform == "linux":
            cpu = subprocess.check_output(["sh", "koblas-bench/select-cpu.sh"], text=True).strip()
            command += ["--cpu", cpu]
        if args.backend != "jvm-simd":
            command += ["--libraries", args.backend]
        if args.native_executable:
            command += ["--native-executable", str(args.native_executable)]
        subprocess.run(command, check=True, env=capture_env)
    paired, provenance, shapes, providers = measurements(results, baseline, candidate, args.backend)
    if not paired:
        raise ValueError("no complete paired measurements")
    header[5] = measured_cpu(results, args.cpu)
    rules = fit(paired, args.backend, shapes)
    digest = hashlib.sha256(json.dumps([header[2:6], rules, sorted(provenance.values())], sort_keys=True).encode()).hexdigest()[:12]
    header[1] = f"{args.runtime}-{args.backend}-{digest}"
    write_profile(args.output / "profile.tsv", rules)
    (args.output / "evidence.json").write_text(json.dumps(dict(
        baseline=baseline, candidate=candidate, source_hashes=provenance,
        paired_ratios=paired, selected_providers=providers, initial_rules=rules, criterion="median <= 0.95 and every round <= 1.02",
    ), indent=2) + "\n")
    if args.capture:
        validation_cases = args.output / "validation-cases.txt"
        training = set(args.cases.read_text().splitlines())
        holdout = [line for line in Path(args.validation_cases).read_text().splitlines()
                   if line.strip() and not line.startswith("#") and line not in training]
        if not holdout:
            raise ValueError("validation requires cases disjoint from training")
        validation_cases.write_text("\n".join(holdout) + "\n")
        validation = args.output / "validation"
        command[command.index("--targets") + 1] = f"{baseline},{args.runtime}-default"
        command[command.index("--cases") + 1] = str(validation_cases)
        command[command.index("--output") + 1] = str(validation)
        subprocess.run(command, check=True, env=dict(os.environ,
                       KOBLAS_BENCH_ENGINE_PROFILE=str((args.output / "profile.tsv").resolve())))
        ratios, hashes, _, reached = measurements(validation, baseline, f"{args.runtime}-default", strict=True)
        if any(rule[1] != "Panels,Products,Triangular" for rule in rules) and args.backend not in reached.values():
            raise ValueError("holdout did not reach any fitted provider; add cases covering the selected regions")
        regressions = {case: values for case, values in ratios.items()
                       if statistics.median(values) > 1.05 and min(values) > 1.02}
        (args.output / "validation.json").write_text(json.dumps(dict(
            paired_ratios=ratios, source_hashes=hashes, selected_providers=reached,
            selected_regressions={case: values for case, values in regressions.items() if reached[case] == args.backend},
            control_regressions={case: values for case, values in regressions.items() if reached[case] != args.backend}),
            indent=2) + "\n")
        if regressions:
            raise ValueError("combined profile or control regression; inspect retained validation.json")
    print(f"Wrote {len(rules)} rules to {args.output / 'profile.tsv'}; retained evidence is available beside the profile.")


if __name__ == "__main__":
    main()
