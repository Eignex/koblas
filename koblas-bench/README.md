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

Use one command to capture a full report in `koblas-bench/reports/<hardware-sha256>/<run-id>/`.
Each run contains one CSV per implementation and `metadata.txt` with completion status, provenance, toolchain
and hardware details. `cpu.csv` records overall CPU utilization once per second, including a three-second
background baseline. Blank readings mean unavailable. Execution timestamps in `metadata.txt` locate each runner
within the trace, including its build and warmup time. The sampler uses the JDK selected by `JAVA_HOME` or `java`
on `PATH`. Repeated runs never overwrite prior results.

```bash
koblas-bench/capture-report.sh --libraries openblas,onemkl \
  --samples 10 --warmups 5 --target-ms 200 --forks 2
```

The command runs JVM scalar, JVM C, JVM SIMD, native koblas, and the requested vendors. A missing selected library
or engine fails the run. Use `--suite packed` for packed cases only. To make a short trial, add
`--operation gemm --warmups 0 --samples 1 --target-ms 1 --forks 1`.

Compare CSVs from the same run (or compatible runs):

```bash
koblas-bench/tools/compare.sh --mode logical --timing prepacked-compute --require-compatible \
  koblas-bench/reports/<hardware-sha256>/<run-id>/openblas.csv \
  koblas-bench/reports/<hardware-sha256>/<run-id>/jvm-c.csv
```

Use `--mode fixed` for identical packed configurations, or `--mode logical` to compare complete operations
across layouts. Physical strategies remain separate pairs. The example selects prepacked block computation;
raw vendor arithmetic has a different timing boundary and cannot be compared with Koblas raw tiles.
The comparator rejects mismatched timing modes, threads, warmups and timing targets. Source SHAs identify the workload and fixtures.
CSV run and case records identify the source commit, runtime, actual kernel and physical configuration.

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
gemm-block+15x7x31+uniform+packed=4x4+timing=prepacked-compute
```

The CSV stores run metadata and case definitions once, followed by sample records referencing their IDs.
Runtime/build strings and source commits belong to the run. Case records contain the case, status, comparison kind,
timing and kernel; shape and packing are read from the case instead of duplicated as metadata.
JMH elapsed time is reconstructed from its score; Native/vendor elapsed time is measured. Keep raw sample values unchanged.
Unsupported cases have no timing; a supported call failure stops the run. Fixtures are deterministic and verified
before relevant runs.

Packed cases choose `packed=4x4` or `packed=8x4`; the recipe fixes layout, strides, zero padding and
alignment independently of backend defaults. Shapes determine panel dimensions, and options may appear in any
order. Blocks require a timing choice. Logical fixtures are generated before packing. `prepacked-compute`
includes tile loops, edge handling and writeback; `pack-plus-compute` also includes both panel packs.

The `scal` and `spgather` cases also accept `+timing=arithmetic`. Scaling then uses `alpha = -1` to
preserve fixture magnitudes across repeated calls; the default resets the vector and uses `alpha = 0.875`.
Arithmetic gather overwrites the sparse values without resetting either buffer. Default gather resets only
the dense source, matching oneMKL. Both runners consume the first and last output values. Compare each timing
boundary separately; the default gather boundary predating this change included an extra Koblas output reset.
OpenBLAS gather remains unsupported because the runner has no corresponding vendor entry point.

## Sparse slices with caller-owned scratch

[`sparse-slices-cases.txt`](sparse-slices-cases.txt) selects 26 cases from the canonical workload. They use
`+timing=reuse`, persistent buffers, and versioned timing labels separate from historical slice cases.
Sorted and shuffled supports have identical paired values; compaction variants retain or discard exact zeros.
The C reference uses the same deterministic fixtures, zero-offset windows, validation scans and output contracts.

| Operation | Timed work | oneMKL reference |
| --- | --- | --- |
| `sparse-slices-cycle` | Two scatters, then gather and clear, optionally compacting zeros | Indexed AXPY and gather-zero plus validation and bookkeeping |
| `sparse-slices-cycle-checked` | Same cycle with latched arithmetic diagnostics | Unsupported; scalar C reference |
| `sparse-slices-gather` | Validate and emit touched indices and values | Indexed gather plus validation and bookkeeping |
| `sparse-slices-gather-clear` | Refill touched entries, then validate, gather and clear | Indexed gather-zero plus validation and bookkeeping |
| `sparse-slices-clear` | Refill touched entries, then validate and clear | Unsupported; scalar C reference |
| `sparse-slices-reduce-dot-checked` | Ordered products and sum with per-step diagnostics | Unsupported; scalar C reference |
| `sparse-slices-reduce-dot-unchecked` | Validated sparse dot through the selected engine | Indexed dot with validation; reduction order may differ |

Each cycle scatters `alpha = 0.875` over the first half of the support, then `alpha = -0.875` over all of it.
The second scatter visits both existing and fresh entries, and exact cancellation exercises compaction.
Gather-clear restores scratch for the next cycle. Standalone gather reads a stable accumulator; standalone
clear and gather-clear include only a touched-entry refill. No case clears the full dimension inside timing.

The oneMKL adapter stages rounded products in reusable scratch before indexed AXPY, preserving separate
multiplication and addition when a vendor implementation uses FMA. Compacted gathers also use reusable scratch
so unwritten output tails remain unchanged. Validation, support marks, touched order, output indices and
compaction all stay inside timing. These are composed comparisons, not raw BLAS timings. Checked operations
remain unsupported by oneMKL because its entry points do not provide the ordered diagnostics contract.

`scalar-slices` is an independent C implementation of all seven contracts. Its runner links the usual OpenBLAS
dependency, but these timed paths make no BLAS calls. Both C references disable floating-point contraction.
Their state checks run before measurement and cover repeated reuse, untouched buffers, cancellation and
exceptional values. JVM tests also check allocation-free reuse. Kotlin rows identify `portable-sparse-slices`
as the actual kernel, except unchecked dot, which uses the selected engine.

Capture a first comparison, optionally pinning the command to a quiet core on Linux:

```bash
koblas-bench/capture-report.sh --suite sparse-slices --libraries scalar-slices,onemkl \
  --warmups 5 --samples 3 --target-ms 100 --forks 2
```

The capture saves its selected `cases.txt` and launches fresh Gradle processes so JVM forks inherit CPU affinity.
Use longer samples and repeat runs before choosing optimization thresholds. For reference-only runs:

```bash
koblas-bench/reference.sh --libraries scalar-slices,onemkl \
  --cases koblas-bench/sparse-slices-cases.txt --output results/sparse-slices
koblas-bench/tools/compare.sh --mode logical --require-compatible \
  results/sparse-slices/onemkl.csv results/sparse-slices/scalar-slices.csv
```

This first suite covers 19 oneMKL compositions and seven scalar-only cases. Other slice operations, including
max and filtering, retain their historical timing boundaries and do not yet have equivalent vendor comparisons.

## Verify the harness

```bash
koblas-bench/reference/test.sh
./gradlew :koblas-bench:jvmTest
./gradlew :koblas-bench:jvmCBenchmark -Pbench.operation=gemm -Pbench.warmups=0 -Pbench.samples=1 -Pbench.targetMs=1 -Pbench.forks=1
./gradlew :koblas-bench:nativeBenchmark -Pbench.operation=dot -Pbench.samples=1 -Pbench.targetMs=1
```

[`example.csv`](example.csv) is a short format example; use complete captured reports for performance comparisons.

The [dense iamax investigation](reports/9096bd06b8f87c06e1a8b7d05912c2b124b9997dd49a7894a304c4ae5c857247/20260912T091639Z-22929a90abb1/README.md) records the scalar bottleneck, kernel design, crossover measurements, and CPU traces.

The [scal and spgather investigation](reports/9096bd06b8f87c06e1a8b7d05912c2b124b9997dd49a7894a304c4ae5c857247/20260912T124303Z-88a37100/README.md) compares both operations with oneMKL and OpenBLAS, records the gather dispatch fix, and documents the scaling experiments.

The [native scaling alignment follow-up](reports/9096bd06b8f87c06e1a8b7d05912c2b124b9997dd49a7894a304c4ae5c857247/20260912T134906Z-scal-alignment/README.md) isolates misaligned stores, validates guarded alignment through the native engine, and compares with oneMKL and OpenBLAS.

The [OpenBLAS source follow-up](reports/9096bd06b8f87c06e1a8b7d05912c2b124b9997dd49a7894a304c4ae5c857247/20260912T141014Z-openblas-source/README.md) compares the Haswell loop, validates fixed pointer-relative blocks, and records the remaining vendor gap.

The [SparseSlices harness validation](reports/9096bd06b8f87c06e1a8b7d05912c2b124b9997dd49a7894a304c4ae5c857247/20260912T161309Z-0081264d482f/README.md) records the first complete reuse-suite capture and supported comparison pairs; background load prevents performance conclusions.
