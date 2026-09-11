# koblas-bench

Benchmarks for koblas, plus independent OpenBLAS and oneMKL reference runs. This module is for development; it is
not published.

## Quick start

Run the koblas implementations you want to compare. `all` is the complete workload; replace it with an operation
such as `gemm` for a smaller run.

```bash
./gradlew :koblas-bench:jvmScalarBenchmark -Pbench.operation=all
./gradlew :koblas-bench:jvmCBenchmark -Pbench.operation=all
./gradlew :koblas-bench:jvmSimdBenchmark -Pbench.operation=all
./gradlew :koblas-bench:nativeBenchmark -Pbench.operation=all
```

Output defaults to `koblas-bench/build/benchmarks/`. Use `-Pbench.output=results/jvm-c.csv` to choose a file.
JVM runs use JDK 25 and JMH; native runs work on Linux x86-64 and macOS arm64. Every run selects the requested
engine exactly, or fails—there is no silent substitute.

Before a long vendor run, check that the library works:

```bash
koblas-bench/reference-smoke.sh --libraries openblas
```

Then run the full vendor workload into a fresh directory:

```bash
koblas-bench/reference.sh --libraries openblas,onemkl --output results/vendor
```

This writes one file per library (`openblas.csv`, `onemkl.csv`). Vendor runs use one thread. OpenBLAS must be
linkable as `-lopenblas`. For oneMKL, set `ONEMKL_LIBRARY=/path/to/libmkl_rt.so.3` if the default runtime path is
not suitable.

## Save and compare results

Use one command to capture a full report. It hashes a stable hardware fingerprint, saves it at
`koblas-bench/reports/<hardware-sha256>/hardware.txt`, then saves this run in a timestamped subdirectory. Repeated
runs on the same hardware share the fingerprint and never overwrite prior results.

```bash
koblas-bench/capture-report.sh --libraries openblas,onemkl
```

The command runs JVM scalar, JVM C, JVM SIMD, native koblas, and the requested vendors. A missing selected library
or engine fails the run. Partial reports and logs remain on disk with `status.txt=incomplete`; a successful
run marks it `complete`. The report keeps an exact case snapshot, source patch, compiler/runtime provenance,
and raw per-run logs. To make a short trial, add
`--operation gemm --warmups 0 --samples 1 --target-ms 1 --forks 1`.

Compare CSVs from the same run (or compatible runs):

```bash
koblas-bench/tools/compare.sh --mode logical --timing prepacked-compute --require-compatible \
  koblas-bench/reports/<hardware-sha256>/<run-id>/vendor/openblas.csv \
  koblas-bench/reports/<hardware-sha256>/<run-id>/jvm-c.csv
```

The example selects the complete prepacked block boundary. Without `--timing`, full vendor reports also
contain intentionally incompatible raw-tile boundaries: omit `--require-compatible` to inspect accepted pairs
and rejection diagnostics together. Use `capture-report.sh --suite packed` to recapture every explicit packed
case from the authoritative workload.

Choose `--mode fixed` for identical packed configurations or `--mode logical` for complete logical workloads.
The comparator refuses mismatched workload/fixture versions, timing modes, threads, warmups, or timing targets.
Physical strategies remain separate pairs; policy rows belong to logical mode. It does not probe your hardware or libraries. `hardware.txt` contains static CPU topology, model, cache,
and memory information only, so its hash does not change due to collection time, kernel version, CPU frequency,
or compiler upgrades. Each CSV row records source commit, dirty state, runtime, thread count, and timing settings.

## Useful options

```text
-Pbench.warmups=3     warmup iterations
-Pbench.samples=5     measured samples
-Pbench.targetMs=1000 target duration per sample
-Pbench.forks=2       JVM forks only
-Pbench.pass=1        label for repeated runs
-Pbench.output=file   CSV destination
```

Every raw measured sample is kept. A busy machine can make results noisy, so avoid comparing runs from different
host conditions when possible.

## Workload and output

[`cases.txt`](cases.txt) is the authoritative workload. Cases look like:

```text
operation+dimensions+fixture[+option=value...]
gemm+129x31x257+uniform+transA=T
spgemv+257x129+sparse-uniform+density=0.01+mode=prepared
```

Do not hand-edit generated CSVs. They use schema 4 and retain provenance, timing, and compatibility metadata.
Unsupported cases have no timing; a supported call failure stops the run. See [`coverage.md`](coverage.md) for
the exact vendor-operation mapping and timing boundaries. Fixtures are deterministic and verified before relevant
runs. See [packed-cases.md](packed-cases.md) for mandatory packed options, immutable formulas, mathematical
identity and timing boundaries. [contracts.md](contracts.md) inventories the semantic baseline and oracle owners.

## Verify the harness

```bash
koblas-bench/reference/test.sh
./gradlew :koblas-bench:jvmTest
./gradlew :koblas-bench:jvmCBenchmark -Pbench.operation=gemm -Pbench.warmups=0 -Pbench.samples=1 -Pbench.targetMs=1 -Pbench.forks=1
./gradlew :koblas-bench:nativeBenchmark -Pbench.operation=dot -Pbench.samples=1 -Pbench.targetMs=1
```

[`example.csv`](example.csv) shows the format only; its smoke timings are not performance evidence.
