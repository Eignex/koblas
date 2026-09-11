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
or engine fails the run. Use `--suite packed` for packed cases only. To make a short trial, add
`--operation gemm --warmups 0 --samples 1 --target-ms 1 --forks 1`.

Compare CSVs from the same run (or compatible runs):

```bash
koblas-bench/tools/compare.sh --mode logical --timing prepacked-compute --require-compatible \
  koblas-bench/reports/<hardware-sha256>/<run-id>/vendor/openblas.csv \
  koblas-bench/reports/<hardware-sha256>/<run-id>/jvm-c.csv
```

Use `--mode fixed` for identical packed configurations, or `--mode logical` to compare complete operations
across layouts. Physical strategies remain separate pairs. The example selects prepacked block computation;
raw vendor arithmetic has a different timing boundary and cannot be compared with Koblas raw tiles.
The comparator rejects mismatched workload/fixture versions, timing modes, threads, warmups and timing targets.
Each CSV row records its source commit, runtime, actual kernel and physical configuration.

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
runs.

Packed cases explicitly declare their tile, versioned layouts, packing settings and timing in `cases.txt`;
unsupported configurations are rejected. Logical fixtures are generated before packing. `prepacked-compute`
includes tile loops, edge handling and writeback; `pack-plus-compute` also includes both panel packs.

## Verify the harness

```bash
koblas-bench/reference/test.sh
./gradlew :koblas-bench:jvmTest
./gradlew :koblas-bench:jvmCBenchmark -Pbench.operation=gemm -Pbench.warmups=0 -Pbench.samples=1 -Pbench.targetMs=1 -Pbench.forks=1
./gradlew :koblas-bench:nativeBenchmark -Pbench.operation=dot -Pbench.samples=1 -Pbench.targetMs=1
```

[`example.csv`](example.csv) shows the format only; its smoke timings are not performance evidence.
