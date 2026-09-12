# Native scaling alignment follow-up

Aligning the C scaling loop for x86 vectors of at least 128 elements fixes a reproducible bottleneck.
The native engine diagnostic improves misaligned n=4096 arithmetic by a median paired **1.87x**
(range 1.80–1.91x), from a pooled median 394.2 to 212.4 ns of thread CPU time. Misaligned n=256
improves 1.56x, and n=65536 improves 1.09x. The default reset-and-arithmetic workload improves 1.23x
at n=4096. These are controlled diagnostic measurements, not JMH or public API performance guarantees.

This follows the [earlier scal and spgather investigation](../20260912T124303Z-88a37100/README.md).
That report correctly rejected alignment based on inconclusive elapsed-time comparisons with uncontrolled
array addresses. This follow-up records the actual alignment, rotates implementations, and preserves
both thread CPU time and elapsed time. The old captures remain unchanged.

## Cause and implementation

The original native AVX2 clone already processes four vectors per iteration. Missing AVX2 support and
insufficient source-level unrolling were not the main problem. An eight-byte-misaligned 32-byte store
stream crosses cache-line boundaries; varying the data address in the C probe roughly doubles its cost
at n=4096. OpenBLAS shows the same sensitivity on this i9-12900H; oneMKL largely avoids it.

The retained C change peels at most three doubles, tells the compiler that the remaining pointer is
32-byte aligned, and leaves the remaining loop to autovectorization. It applies only to x86-64 and
lengths of at least 128. Empty calls and alpha=1 return immediately. Short calls and other architectures
use the original multiplication loop. Multiplication semantics, including NaNs and signed zeros, remain
those of the scalar oracle. The native binary's large-vector AVX2 loop uses aligned stores; see the
before/after assembly files.

At n=64 unconditional peeling loses time, so it was rejected. The cutoff probe covers n=96, 128, 192,
and 256, with offsets 0, 8, 16, and 24 bytes and three coefficient/reset modes. A conservative cutoff of
128 retains the measured improvement. Explicit eight-vector unrolling was also tested; its small extra
gain at n=4096 did not justify adding that complexity.

The aligned n=4096 native arithmetic case has a small regression: median paired speedup 0.97x
(range 0.96–0.98x), with pooled times 215.2 versus 219.7 ns. The much larger misaligned improvement is
the reason to retain the guard and peeling. n=64 is effectively unchanged. Results are specific to this
AVX2 host; this does not establish a gain on every x86 processor. JVM scaling is unchanged, including
the JVM C engine's Kotlin scalar scaling path.

## Comparison with oneMKL and OpenBLAS

The final C diagnostic rotates the original C leaf, guarded alignment candidate, two unrolling candidates,
OpenBLAS, and oneMKL in one process. It loads the vendors separately with local, deep symbol binding and
sets each to one thread. All arms operate on the same pointer within a case.

For misaligned n=4096 negation, the guarded C candidate is about 193 ns versus oneMKL's 165 ns and
OpenBLAS's 403 ns: about 17% more time than oneMKL, and about 2.1x faster than OpenBLAS. At n=65536
the candidate is close to oneMKL. These leaf measurements exclude Kotlin pinning and dispatch; do not
combine them with the native engine timings to claim exact end-to-end vendor ratios.

The standard suite, whose arrays have uncontrolled alignment, shows little before/after change in this
capture. At n=4096 its new native arithmetic median is 230.0 ns versus 208.1 ns for oneMKL and 204.1 ns
for OpenBLAS. At n=65536 it is 7,556.5 ns versus 7,604.1 ns and 8,275.1 ns respectively. These numbers
include each harness's normal boundary and do not replace the controlled offset evidence above.

The [tables](tables.md) include all three boundaries (negation, alternating exact powers of two, and
reset plus multiplication), aligned/misaligned native results, and the standard benchmark suite's
separate elapsed-time vendor comparison. Reset measurements include copying and must not be subtracted
from negation timings as an exact estimate of copy cost.

## Measurement method and provenance

The C diagnostics use Clang 21 from the Kotlin/Native toolchain, -O3, default/AVX2 target clones, CPU 4,
and one thread per vendor. Their fixture is binary-exact `(i % 251 - 125) / 256`, kept finite by negation
or alternating 0.5 and 2.0; reset mode copies the initial vector then multiplies by 0.875. This is a
diagnostic fixture, distinct from the standard suite's deterministic uniform fixture. Each C sample
rotates six arms over 32 rounds; the first two samples are warmups, then seven samples are retained
(five for the cutoff probe). Short-batch clock overhead is included equally in each arm.

`ScalDiagnostic.kt` calls the actual native C engine with the standard `Fixtures.vector(n, 1)` fixture,
including pinning, offset handling, reset where requested, and primitive first/last result consumption.
Each sample has 16 batches, with 1,048,576/n calls per batch. Three warmup samples precede seven measured
samples. The same backing array is exercised at four offsets, with the actual pointer modulo 32 recorded.
Three process passes alternate before/after, after/before, before/after. Both binaries use identical
diagnostic code and differ in the production scaling implementation.

Both diagnostics record CLOCK_THREAD_CPUTIME_ID and CLOCK_MONOTONIC. They use their own explicit CSV
schema and do not change the standard suite's clock or label CPU time as ordinary suite elapsed time.
In this capture CPU and elapsed times are close: the guarded C sample median elapsed/CPU ratio is 1.006.
The stronger evidence here comes from controlling alignment and run order; these data do not prove that
scheduler delays caused every earlier inconsistent measurement. CPU clocks still include contention
and frequency effects while the thread runs.

- Baseline production source for the C and native diagnostics: `204f8d29742e70178056b24801d91322d932569f`.
- Candidate production source: `ef35f7ca` (the diagnostic binaries were built with temporary entry-point
  and source additions, removed before committing; no production kernel difference from that commit).
- Standard-suite old native binary: `5e5a7f1f1de41b1824299822f66314b6af1574ef`, retained from the earlier
  alignment trial. Its scaling implementation and scal workload match the diagnostic baseline.
- Standard-suite new native binary and vendor harness: `ef35f7ca`. Raw rows contain full source IDs.
  The run metadata's dirty flag was corrected to true after discovering that Gradle had formatted two
  test files before the capture. Production and harness sources still match the recorded commit;
  sample rows are unchanged.
- `probe.c` / `probe.csv`: original unguarded alignment and unrolling experiments.
- `cutoff.c` / `cutoff.csv`: short-size cutoff experiment.
- `guarded.c` / `guarded.csv`: final guarded candidate versus the original and vendors.
- `native-before-*.csv` and `native-after-*.csv`: actual engine diagnostics.
- `suite/`: canonical cases, alternating old/new/vendor samples, logs, timestamps and host CPU trace.

The diagnostic C sources include the original header, so reproduce them against the baseline checkout,
not a current header whose scale function has already changed. The guarded candidate body is embedded
in `guarded.c`. `reproduce-diagnostics.sh` handles the two revisions in temporary worktrees and records
new results in a fresh output directory. `capture-suite.sh` preserves the exact commands used for the
standard-suite capture; its /tmp paths identify the captured binaries and should be adjusted when rerunning.
Run `python3 summarize-results.py` to regenerate the tables from this report's raw samples.

## Validation

Passed `./gradlew :koblas:check :koblas-bench:check lintDocs check` and
`./gradlew :koblas:jvmTest -Pkoblas.noSimd=true`. New scalar-oracle conformance coverage checks offsets,
cutoff boundaries, vector tails, unchanged prefixes/suffixes, signed zeros, subnormals, infinities,
NaNs and alpha=1. Its JVM test takes 40 ms. Native conformance executes the actual C leaf.
