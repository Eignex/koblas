# Apple M4 full CPU benchmark

The complete 175-case workload at source commit `c344bccbeda7f52d0e683eada1c06daee7c3d22a`
ran successfully across JVM scalar, JVM C, JVM SIMD, Kotlin/Native C, OpenBLAS, and Apple Accelerate.
The capture ran from 13:46:05 to 14:41:18 UTC on September 12, 2026. No GPU backend was used.
oneMKL was intentionally omitted on this macOS arm64 host.

## Reproduce

Apple M4, 10 physical CPUs, 24 GiB RAM, macOS 26.6.2 (25G83), Homebrew JDK 25.0.4.1,
Kotlin 2.4.10, JMH 1.37, and Apple Clang 21.0.0. OpenBLAS 0.3.34 reports its `vortexm4`
dynamic architecture. Accelerate is the system framework. JVM SIMD reports two double lanes.

Run from the repository root at the source commit:

```bash
caffeinate -i koblas-bench/capture-report.sh --libraries openblas,accelerate \
  --samples 10 --warmups 5 --target-ms 200 --forks 2
```

All runners use one benchmark thread. Vendor thread controls are set by `reference.sh`.
The JVM uses two forks with ten measured iterations each. Native and vendor runners use one process
and ten measured samples. All have five warmup iterations, but JVM warmups target 200 ms and
native/vendor warmups target 50 ms. Calibrated runners cap their batch size, so short operations can
have measured samples shorter than the 200 ms target. JVM elapsed times are reconstructed from JMH scores.

The recorded `dirty=true` is caused by the newly created, untracked report directory itself.
Source changes were committed before capture; there is no source patch. The workload is pinned by
the recorded source commit. Subsequent dense `iamax` changes and six additional cases on `main`
are not represented by this capture.

## Coverage and integrity

Every implementation records all 175 case statuses. Unsupported cases have no fabricated timings.

| Implementation | Measured cases | Unsupported cases | Raw samples |
| --- | ---: | ---: | ---: |
| JVM scalar | 133 | 42 | 2,660 |
| JVM C | 133 | 42 | 2,660 |
| JVM SIMD | 133 | 42 | 2,660 |
| Native C | 133 | 42 | 1,330 |
| OpenBLAS | 72 | 103 | 720 |
| Accelerate | 78 | 97 | 780 |

The 10,810 raw samples were checked for finite positive timings, valid operation counts, unique
case/fork/sample identifiers, expected per-fork sample counts, matching source commits and timing
settings, and exact coverage of the source workload. `metadata.txt` records `status=complete`.
Vendor numerical preflight checks passed before timing. OpenBLAS has no sparse coverage in this
runner. Accelerate adds sparse dot, AXPY, norms, matrix-vector, and matrix-times-dense-matrix coverage;
its other sparse operations remain explicitly unsupported. Using an optimized CPU framework does
not establish that every individual sparse call executes SIMD instructions.

## Selected results

Medians in ns/op; lower is better. JVM medians combine all twenty samples, other medians use ten.
These are the exact named case records, without pooling different packed recipes.
All sparse fixtures below use density 0.01. Sparse vectors have logical length 4096.
GEMM dimensions are rows by output columns by inner dimension.

| Case | JVM scalar | JVM C | JVM SIMD | Native C | OpenBLAS | Accelerate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dot, 4096 | 2,145.67 | 371.65 | 442.47 | 370.95 | 605.21 | 147.62 |
| gemm, 32x21x48 | 6,604.40 | 4,326.94 | 4,172.88 | 6,528.19 | 1,519.44 | 902.05 |
| gemm, 129x31x257, transA=T | 167,548.41 | 98,690.42 | 96,712.99 | 133,184.02 | 49,166.50 | 12,230.90 |
| gemm-block, 64x64x64, packed=4x4, prepacked-compute | 40,172.80 | 25,811.28 | 22,960.25 | 47,973.72 | 10,706.27 | 2,743.87 |
| gemm-block, 128x128x64, packed=4x4, prepacked-compute | 159,867.13 | 103,192.81 | 89,578.01 | 192,661.83 | 40,849.24 | 8,338.71 |
| spdot | 12.44 | 12.41 | 12.37 | 31.53 | — | 21.34 |
| spaxpy | 654.02 | 721.18 | 705.81 | 600.01 | — | 585.61 |
| spnrm2 | 7.25 | 7.37 | 5.75 | 15.10 | — | 14.86 |
| spasum | 6.46 | 6.43 | 5.32 | 14.93 | — | 12.77 |
| spgemv, 257x129, prepared | 319.75 | 324.91 | 323.44 | 887.08 | — | 1,131.81 |
| spgemv, 257x129, oneshot | 330.58 | 332.92 | 323.61 | 838.07 | — | 14,495.42 |
| spmm, 257x8x129, prepared | 2,256.65 | 2,248.79 | 3,188.57 | 2,973.65 | — | 5,017.49 |
| spmm, 257x8x129, oneshot | 2,339.74 | 2,253.63 | 3,262.54 | 3,078.32 | — | 18,659.97 |

Accelerate is fastest for these selected dense cases. The sparse result depends on the operation:
JVM Koblas leads dot, norms, and the matrix cases here, while Accelerate has the lowest sparse AXPY
median. SIMD selection is not universally faster: prepared sparse matrix-times-dense-matrix is
about 1.42 times slower in the JVM SIMD arm than in JVM C for this fixture.

Dot and norms time arithmetic. Dense GEMM and sparse AXPY include output reset. Sparse matrix
cases also reset output; Accelerate scales the destination by beta before its additive sparse call.
Prepared Accelerate matrices are built and committed before timing. One-shot timings include matrix
creation, entry insertion, commit, and destruction. The large one-shot/prepared gap therefore includes
representation setup, not just multiplication. Packed block comparisons match complete logical
operations: Koblas consumes packed panels while vendors consume column-major input. They are not
comparisons of identical internal kernels or raw tiles.

This is a single host run, not a confidence interval or a crossover study. The machine was not
isolated or core-pinned. Overall CPU utilization was about 10.6% at the captured baseline; mean
utilization during scalar, C, SIMD, native, and combined vendor intervals was 14.0%, 13.3%, 13.3%,
12.8%, and 12.5%, respectively. These include background work, builds, and warmups and are not
per-benchmark CPU measurements. See `cpu.csv` and execution timestamps before drawing tuning conclusions.

## Machine-readable comparisons

`comparison-openblas.csv` and `comparison-accelerate.csv` contain compatible logical comparisons
against all four Koblas implementations. `base_over_candidate` greater than one means Koblas is faster.
Raw tile and packing-only boundaries are excluded. Logical comparisons pool equivalent vendor
column-major records across packed recipe labels; the table above instead selects exact case records.
Regenerate from this directory, replacing `accelerate` with `openblas` for the other file:

```bash
../../../tools/compare.sh --mode logical --require-compatible \
  --timing arithmetic --timing reset-and-arithmetic --timing prepared \
  --timing oneshot --timing prepacked-compute \
  accelerate.csv jvm-scalar.csv jvm-c.csv jvm-simd.csv native.csv
```

## Fixes and validation

The pre-capture allocation regression test exposed 22,528 allocated bytes per indexed SIMD dot call
on this NEON/JDK 25 combination. Replacing the two-lane indexed load with inlined scalar lane loads
made the existing allocation test pass its 64-byte-per-call ceiling. Scalar-oracle conformance checks
also passed. Wider SIMD species retain the existing indexed-load path. This report measures the fixed
implementation; it is not a before/after performance comparison.

The harness now locates Homebrew's keg-only OpenBLAS, supports Accelerate, and classifies Accelerate
packed comparisons as column-major. Sparse norm comparison metadata and macOS comparator tests were
corrected. Accelerate tests check nonzero dot fixtures, every supported sparse operation, rectangular
matrix dimensions, complete outputs, repeat-call resets, and prepared/one-shot lifetimes against scalar results.

Successful validation before capture:

```bash
koblas-bench/reference/test.sh
./gradlew :koblas:check :koblas-bench:check lintDocs
./gradlew check
```

All-case short preflight runs also succeeded for C, SIMD, native, OpenBLAS, and Accelerate before
the full capture. Raw CSVs and CPU trace are preserved unchanged.
