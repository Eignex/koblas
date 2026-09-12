# koblas-bench

Benchmarks for koblas, plus independent OpenBLAS, Accelerate, and oneMKL reference runs. This module is for development; it is
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
koblas-bench/reference.sh --libraries all --output results/vendor
```

This writes one file per library. On macOS, `all` selects OpenBLAS and Accelerate; other hosts select OpenBLAS and
oneMKL. Vendor runs use one thread. OpenBLAS must be linkable as `-lopenblas`; on macOS, the runner automatically
uses Homebrew's keg-only `openblas` install. Accelerate is the system comparator for macOS and covers dense BLAS,
sparse vector dot/AXPY/norms, sparse matrix-vector multiplication, and sparse matrix times dense matrix multiplication.
Accelerate requires macOS 15 or later for single-thread BLAS control; `VECLIB_MAXIMUM_THREADS=1` also limits its sparse calls.
Prepared sparse cases commit the matrix before timing; one-shot cases include creation, insertion, commit, and destruction.
For oneMKL, set
`ONEMKL_LIBRARY=/path/to/libmkl_rt.so.3` if the default runtime path is not suitable. oneMKL vendor runs require a
supported Linux runtime.

## Save and compare results

Use one command to capture a full report in `koblas-bench/reports/<hardware-sha256>/<run-id>/`.
Each run contains one `*-summary.csv` per implementation: one row per case with sample count, fork count,
median, minimum, and maximum ns/op. Run provenance is stored once. `cpu-summary.csv` has one row per observed runner
phase plus the background baseline, with reading counts and mean, median, minimum, and maximum CPU usage.
`metadata.txt` records completion status, provenance, toolchain, hardware, and execution timestamps.
Blank statistics mean unavailable, not zero. CPU phases include builds and warmups, not just arithmetic.

Runners write summaries directly; there is no raw CSV export or separate summarization step. CPU usage
is sampled once per second and accumulated in memory. Phase boundaries are observed on those ticks,
so phases shorter than the sampling interval may have no CPU row. The Kotlin CPU sampler uses the JDK
selected by `JAVA_HOME` or `java` on `PATH`. Repeated runs never overwrite prior results.

```bash
koblas-bench/capture-report.sh --libraries all \
  --samples 10 --warmups 5 --target-ms 200 --forks 2
```

The command runs JVM scalar, JVM C, JVM SIMD, native koblas, and the requested vendors. A missing selected library
or engine fails the run. Use `--suite packed` for packed cases only. To make a short trial, add
`--operation gemm --warmups 0 --samples 1 --target-ms 1 --forks 1`.

Compare CSVs from the same run (or compatible runs):

```bash
koblas-bench/tools/compare.sh --mode logical --timing prepacked-compute --require-compatible \
  koblas-bench/reports/<hardware-sha256>/<run-id>/openblas-summary.csv \
  koblas-bench/reports/<hardware-sha256>/<run-id>/jvm-c-summary.csv
```

Use `--mode fixed` for identical packed configurations, or `--mode logical` to compare complete operations
across layouts. Physical strategies remain separate pairs. The example selects prepacked block computation;
raw vendor arithmetic has a different timing boundary and cannot be compared with Koblas raw tiles.
The comparator requires GNU awk (`gawk`; install with `brew install gawk` on macOS).
It rejects mismatched timing modes, threads, warmups and timing targets. Source SHAs identify the workload and fixtures.
CSV run and case records identify the source commit, runtime, actual kernel and physical configuration.
The comparator also accepts legacy raw captures. Compact case records remain separate: summary medians
cannot reconstruct a pooled sample distribution, even when different cases represent equivalent work.

## Useful options

```text
-Pbench.warmups=3     warmup iterations
-Pbench.samples=5     measured samples
-Pbench.targetMs=1000 target duration per sample
-Pbench.forks=2       JVM forks only
-Pbench.pass=1        label for repeated runs
-Pbench.output=file   CSV destination
```

All measured samples contribute to the summary. A busy machine can make results noisy, so avoid comparing runs from different
host conditions when possible.

## Workload and output

[`cases.txt`](cases.txt) is the authoritative workload. Cases look like:

```text
operation+dimensions+fixture[+option=value...]
gemm+129x31x257+uniform+transA=T
spgemv+257x129+sparse-uniform+density=0.01+mode=prepared
gemm-block+15x7x31+uniform+packed=4x4+timing=prepacked-compute
```

The CSV stores run metadata once and one summary row per case.
Runtime/build strings and source commits belong to the run. Case records contain the case, status, comparison kind,
timing and kernel; shape and packing are read from the case instead of duplicated as metadata.
Case rows include measured sample and fork counts, median, minimum, and maximum ns/op.
JVM statistics combine all JMH measured iterations across forks; Native/vendor timings use measured elapsed time.
Unsupported cases have no timing; a supported call failure stops the run. Fixtures are deterministic and verified
before relevant runs.

Packed cases choose `packed=4x4` or `packed=8x4`; the recipe fixes layout, strides, zero padding and
alignment independently of backend defaults. Shapes determine panel dimensions, and options may appear in any
order. Blocks require a timing choice. Logical fixtures are generated before packing. `prepacked-compute`
includes tile loops, edge handling and writeback; `pack-plus-compute` also includes both panel packs.

## Verify the harness

```bash
koblas-bench/reference/test.sh
./gradlew :koblas-bench:jvmTest
./gradlew :koblas-bench:jvmCBenchmark -Pbench.operation=gemm -Pbench.warmups=0 -Pbench.samples=1 -Pbench.targetMs=1 -Pbench.forks=1
./gradlew :koblas-bench:nativeBenchmark -Pbench.operation=dot -Pbench.samples=1 -Pbench.targetMs=1
```

[`example.csv`](example.csv) is a short format example; use complete captured reports for performance comparisons.
