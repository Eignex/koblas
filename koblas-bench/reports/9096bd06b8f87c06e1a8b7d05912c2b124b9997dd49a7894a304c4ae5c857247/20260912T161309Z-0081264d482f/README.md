# SparseSlices comparison harness validation

This capture validates the new caller-owned scratch comparison suite. It is not suitable for performance
rankings or dispatch thresholds: concurrent builds loaded the host, and the 100 ms windows are short.
Overall CPU utilization had a median of 58.29% and a maximum of 99.20% across the capture, including builds.
See [cpu.csv](cpu.csv) and the execution timestamps in [metadata.txt](metadata.txt).

## Reproduce

Source commit: `0081264d482f` (the full SHA is in the CSV run records). No source edits occurred during capture.
The dirty marker reflects the newly generated, untracked report directory. Linux x86-64, Intel Core i9-12900H,
CPU 4 affinity, one thread per benchmark, JDK 25.0.1, Kotlin 2.4.10 and oneMKL 2026.1.

```bash
taskset -c 4 koblas-bench/capture-report.sh --suite sparse-slices \
  --libraries scalar-slices,onemkl --warmups 5 --samples 3 --target-ms 100 --forks 2
```

The exact selection is saved in [cases.txt](cases.txt). JVM runs use two forks; native and C use one.
JVM warmups target 100 ms each; the existing calibrated native/C harness targets 25 ms per warmup.
All measured samples target 100 ms. Repeat with longer windows on a quiet host before drawing speed conclusions.

## Coverage and matching

| Arm | Supported cases | Unsupported cases | Raw samples |
| --- | ---: | ---: | ---: |
| JVM scalar | 26 | 0 | 156 |
| JVM C | 26 | 0 | 156 |
| JVM SIMD | 26 | 0 | 156 |
| Native | 26 | 0 | 78 |
| Scalar C slices | 26 | 0 | 78 |
| oneMKL compositions | 19 | 7 | 57 |

The strict logical comparator produces 95 pairs in [onemkl-comparisons.csv](onemkl-comparisons.csv):
19 supported cases against each of five other arms. It produces 104 pairs in
[scalar-slices-comparisons.csv](scalar-slices-comparisons.csv): all 26 cases against four Koblas arms.
The ratios are retained for reproducibility, not used to rank implementations under this host load.

Seven operations cover complete scatter/gather-clear cycles, checked cycles, gather, gather-clear, clear,
checked ordered dot and unchecked dot. The 12 unchecked cycle cases vary support size (8, 256, 1024),
sorted/shuffled locality and exact-zero compaction. The other 14 cases cover diagnostics and individual phases.
oneMKL checked-cycle, checked-dot and clear-only rows remain explicitly unsupported.

Kotlin slice compositions record `portable-sparse-slices` as the actual kernel across all engine selections;
only unchecked dot dispatches through the selected engine. oneMKL rows include validation, support tracking,
output indices and compaction. Product staging preserves separate multiply/add rounding; gather scratch
preserves unwritten output tails. No timed call allocates scratch or clears the full dense dimension.
See the [suite contracts](../../../README.md#sparse-slices-with-caller-owned-scratch) for timing boundaries.

## Validation

- Full `./gradlew :koblas:check :koblas-bench:check lintDocs check` passed.
- JVM and Linux native state tests passed, including repeated reuse, cancellation, marks, untouched scratch and output tails.
- JVM allocation checks passed for all seven reusable operations.
- Scalar C and oneMKL state/numerical checks passed before reference measurements, including exceptional values and ordered diagnostics.
- Reference parser and comparator tests passed, including explicit unsupported records.
- Every CSV has the expected case/sample count, and both strict comparator joins passed.
