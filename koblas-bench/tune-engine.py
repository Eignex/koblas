#!/usr/bin/env python3
"""Fit conservative engine preferences from repeated, paired capture-report outputs."""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from tuning_capture import measurements

MAX = 2**31 - 1


def fit(paired, backend, shapes, by_cpu=None, cache_bytes=None):
    groups = {}
    for case, ratios in paired.items():
        key = shapes[case]
        if key:
            groups.setdefault(key, []).extend((by_cpu or {}).get(case, {"local": ratios}).values())
    wins = {key: all(statistics.median(ratios) <= 0.95 and max(ratios) <= 1.02 for ratios in strata)
            for key, strata in groups.items()}
    rules = []
    for operation, contiguous, borrowed in sorted({(key[0], key[6], key[4]) for key in groups}):
        keys = sorted(key for key in groups if (key[0], key[6], key[4]) == (operation, contiguous, borrowed))
        if keys[0][5] == 1:
            # A threshold must keep winning at every larger measured length, including all variants.
            start = next((i for i in range(len(keys)) if all(wins[k] for k in keys[i:])), None)
            if start is not None:
                tail = keys[start:][-2:]
                beyond_cache = cache_bytes is not None and len(tail) == 2 and all(key[1] * 8 > cache_bytes for key in tail)
                rules.append([backend, operation, keys[start][1], MAX if beyond_cache else keys[-1][1],
                              0, MAX, 0, MAX, 0, str(contiguous).lower(), "*"])
        else:
            # Fit work and minimum extents jointly; skinny losing calls constrain the same region.
            dimensions = 3 if keys[0][5] == 3 else 2
            work = lambda key: key[1] * key[2] * key[3]
            extent = lambda key: min(key[1:1 + dimensions])
            regions = []
            for minimum in sorted({extent(key) for key in keys if wins[key]}):
                for threshold in sorted({work(key) for key in keys if wins[key]}):
                    covered = {key for key in keys if extent(key) >= minimum and work(key) >= threshold}
                    if len(covered) >= 2 and all(wins[key] for key in covered):
                        regions.append((len(covered), -minimum, -threshold, covered))
            covered = set()
            if regions:
                _, minimum, threshold, covered = max(regions, key=lambda region: region[:3])
                rules.append([backend, operation, -minimum, MAX, -minimum, MAX,
                              -minimum if dimensions == 3 else 0, MAX, -threshold, "*", str(borrowed).lower()])
            # Residual deep wins become depth bands; losing measured depths split a band.
            for rows, columns in sorted({key[1:3] for key in keys}):
                band = []
                for key in [key for key in keys if key[1:3] == (rows, columns)] + [None]:
                    if key is not None and wins[key] and key not in covered:
                        band.append(key[3])
                    elif band:
                        rules.append([backend, operation, rows, rows, columns, columns,
                                      min(band), max(band), 0, "*", str(borrowed).lower()])
                        band = []
    if backend == "jvm-simd":
        rules.append([backend, "Panels,Products,Triangular", 0, MAX, 0, MAX, 0, MAX, 0, "*", "*"])
    return rules


def compose(captures):
    fitted = {}
    for source, capture in captures.items():
        caches = list(capture.cache_by_cpu.values())
        fitted[source] = fit(capture.paired, source, capture.shapes, capture.by_cpu,
                             max(caches) if caches and all(caches) else None)
    rules, precedence = [], {}
    vendors = {source: rows for source, rows in fitted.items() if source != "jvm-simd"}
    for operation in sorted({rule[1] for rows in vendors.values() for rule in rows}):
        sources = [source for source, rows in vendors.items() if any(rule[1] == operation for rule in rows)]
        common = set.intersection(*({case for case in captures[source].paired
                                    if captures[source].shapes[case] and captures[source].shapes[case][0] == operation} for source in sources))
        costs = {}
        for source in sources:
            by_model = {}
            for case in sorted(common):
                models = set.intersection(*(set(captures[other].by_cpu[case]) for other in sources))
                for model in sorted(models):
                    lengths = {len(captures[other].by_cpu[case][model]) for other in sources}
                    if len(lengths) != 1:
                        raise ValueError("vendor precedence needs the same paired rounds")
                    by_model.setdefault(model, []).append(math.log(statistics.median(captures[source].by_cpu[case][model])))
            if not by_model:
                raise ValueError(f"no paired vendor precedence evidence for {operation}")
            costs[source] = math.exp(statistics.mean(statistics.mean(values) for values in by_model.values()))
        ordered = sorted(sources, key=lambda source: (costs[source], source))
        precedence[operation] = dict(sources=ordered, relative_cost=costs, cases=sorted(common))
        rules.extend(rule for source in ordered for rule in fitted[source] if rule[1] == operation)
    return rules + fitted.get("jvm-simd", []), precedence


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
    parser.add_argument("--runtime", choices=("native", "jvm"))
    parser.add_argument("--backend", action="append", choices=("onemkl", "openblas", "aocl", "armpl", "accelerate", "jvm-simd"),
                        help="repeat to fit a complete composition with measured vendor precedence")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-cases", type=Path, help="write the shared training and holdout grids without measuring")
    parser.add_argument("--candidate", help="capture target for one explicitly configured candidate")
    parser.add_argument("--baseline", choices=("native", "jvm-scalar", "jvm-simd", "jvm-owned"))
    parser.add_argument("--simd-target", default="jvm-simd", help="owned SIMD capture target; JVM profiles include its fitted rules")
    parser.add_argument("--results", type=Path, help="local capture or fetched lab job")
    parser.add_argument("--lab-arm", help="SDK arm in a multi-arm fetched job")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--cases", type=Path, help="custom training cases; defaults to a regular dense grid")
    parser.add_argument("--validation-cases", default="koblas-bench/cases-addressed.txt")
    parser.add_argument("--native-executable", type=Path)
    parser.add_argument("--os", help="require the measured operating system")
    parser.add_argument("--arch", help="require the measured architecture")
    parser.add_argument("--cpu", help="runtime CPU model; use * explicitly for platform wide profiles")
    args = parser.parse_args()
    if args.write_cases:
        args.write_cases.mkdir(parents=True, exist_ok=True)
        training = training_cases()
        (args.write_cases / "training.txt").write_text("\n".join(training) + "\n")
        (args.write_cases / "holdout.txt").write_text(Path(__file__).with_name("cases-addressed.txt").read_text())
        timing = {"dot+64+uniform", "dot+1000000+uniform", "copy+1000000+uniform+stride=2",
                  "axpy+64+uniform+stride=-2", "gemm+128x128x128+uniform", "gemm+16x16x2048+uniform",
                  "trsm+128x128+triangular+side=L+uplo=L+transA=N+diag=N"}
        (args.write_cases / "native-timing.txt").write_text("\n".join(case for case in training if case in timing) + "\n")
        return
    if args.runtime is None or args.backend is None or args.output is None:
        parser.error("--runtime, --backend and --output are required for tuning")
    if args.capture == bool(args.results):
        parser.error("choose --capture or --results")
    if args.capture and args.simd_target != "jvm-simd":
        parser.error("local capture uses the exact jvm-simd arm")
    if "jvm-simd" in args.backend and args.runtime != "jvm":
        parser.error("jvm-simd requires --runtime jvm")
    sources = list(dict.fromkeys(args.backend))
    if args.candidate and len(sources) != 1:
        parser.error("--candidate names one backend; omit it to fit multiple exact arms")
    baseline = args.baseline or ("native" if args.runtime == "native" else "jvm-owned")
    if baseline.startswith("jvm") != (args.runtime == "jvm"):
        parser.error("baseline must use the selected runtime")
    candidates = {source: args.candidate or source + ("-jvm" if args.runtime == "jvm" else "")
                  for source in sources if source != "jvm-simd"}
    if args.runtime == "jvm":
        candidates["jvm-simd"] = args.candidate if sources == ["jvm-simd"] and args.candidate else args.simd_target
    args.output.mkdir(parents=True, exist_ok=False)
    if args.cases is None:
        args.cases = args.output / "training-cases.txt"
        args.cases.write_text("\n".join(training_cases()) + "\n")
    results = args.results or args.output / "capture"
    targets = list(dict.fromkeys((["native"] if args.runtime == "native" else ["jvm-scalar", args.simd_target]) + list(candidates.values())))
    if args.capture:
        command = ["bash", "koblas-bench/capture-report.sh", "--targets", ",".join(targets),
                   "--cases", str(args.cases), "--rounds", "3", "--samples", "5", "--warmups", "20",
                   "--target-ms", "200", "--forks", "2", "--output", str(results)]
        vendors = [source for source in sources if source != "jvm-simd"]
        if vendors:
            command += ["--libraries", ",".join(vendors)]
        if args.native_executable:
            command += ["--native-executable", str(args.native_executable)]
        subprocess.run(command, check=True)
    captures = {source: measurements(results, "jvm-scalar" if source == "jvm-simd" else baseline,
                                    candidate, source, arm=args.lab_arm, owned_target=args.simd_target)
                for source, candidate in candidates.items()}
    if any(not capture.paired for capture in captures.values()):
        raise ValueError("no complete paired measurements for a requested backend")
    target = next(iter(captures.values())).target(args.runtime, args.os, args.arch, args.cpu)
    if any(capture.target(args.runtime, args.os, args.arch, args.cpu) != target for capture in captures.values()):
        raise ValueError("composition sources were measured on different targets")
    rules, precedence = compose(captures)
    hashes = {path: digest for capture in captures.values() for path, digest in capture.hashes.items()}
    digest = hashlib.sha256(json.dumps([target, rules, sorted(hashes.values())], sort_keys=True).encode()).hexdigest()[:12]
    header = ["koblas-profile-v2", f"{args.runtime}-{'-'.join(candidates)}-{digest}", *target,
              str(results), "see evidence.json for paired measurements and hashes"]
    (args.output / "profile.tsv").write_text("\n".join("\t".join(map(str, row)) for row in [header, *rules]) + "\n")
    (args.output / "evidence.json").write_text(json.dumps(dict(
        baseline=baseline, candidates=candidates, source_hashes=hashes, target=target, vendor_precedence=precedence,
        sources={source: dict(paired_ratios=capture.paired, ratios_by_cpu=capture.by_cpu, lab=capture.lab,
                             source_commits=sorted(capture.commits), last_level_cache_bytes=capture.cache_by_cpu,
                             selected_providers=capture.providers) for source, capture in captures.items()},
        rules=rules, criterion="every variant and CPU: median <= 0.95 and every round <= 1.02",
        bounded_vector_rules=[dict(operation=rule[1], last_measured_length=rule[3],
                                   reason="No verified beyond-cache tail; routing remains bounded")
                              for rule in rules if rule[9] != "*" and rule[3] < MAX],
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
        reference = ["jvm-scalar", args.simd_target] if baseline == "jvm-owned" else [baseline]
        command[command.index("--targets") + 1] = ",".join([*reference, f"{args.runtime}-default"])
        command[command.index("--cases") + 1] = str(validation_cases)
        command[command.index("--output") + 1] = str(validation)
        subprocess.run(command, check=True, env=dict(os.environ,
                       KOBLAS_BENCH_ENGINE_PROFILE=str((args.output / "profile.tsv").resolve())))
        checked = measurements(validation, baseline, f"{args.runtime}-default", strict=True, owned_target=args.simd_target)
        reached = {provider for models in checked.providers.values() for provider in models.values()}
        fitted_sources = {rule[0] for rule in rules if rule[1] != "Panels,Products,Triangular"}
        if not fitted_sources.issubset(reached):
            raise ValueError(f"holdout did not reach fitted providers {sorted(fitted_sources - reached)}")
        regressions = {case: models for case, models in checked.by_cpu.items()
                       if any(statistics.median(values) > 1.05 and min(values) > 1.02 for values in models.values())}
        (args.output / "validation.json").write_text(json.dumps(dict(
            paired_ratios=checked.paired, ratios_by_cpu=checked.by_cpu, source_hashes=checked.hashes,
            selected_providers=checked.providers, regressions=regressions), indent=2) + "\n")
        if regressions:
            raise ValueError("combined profile or control regression; inspect retained validation.json")
    print(f"Wrote {len(rules)} rules to {args.output / 'profile.tsv'}; retained evidence is available beside the profile.")


if __name__ == "__main__":
    main()
