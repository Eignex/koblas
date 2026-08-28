# koblas-bench

Development benchmarks for koblas. They use JMH on the JVM and the
kotlinx-benchmark native harness. This module is not published.

## Run a suite

```bash
./gradlew :koblas-bench:jvmLevel3Benchmark
./gradlew :koblas-bench:linuxX64Level3Benchmark
./gradlew :koblas-bench:jvmBenchmark
```

Suite tasks are prefixed with `jvm`, `linuxX64`, or `macosArm64`; only the host
target runs. Results are written under
`koblas-bench/build/reports/benchmarks/<suite>/`.

| Suite | Measures |
| --- | --- |
| `level1` | vector kernels |
| `level2` | matrix-vector kernels |
| `level3` | matrix-matrix kernels |
| `solve` | dense factorization and solves |
| `solveFocused` | the solve rows alone, out to 2048 |
| `blockSolve` | block versus per-column solves |
| `sparse` | sparse factorization, solve, and `gemv` |

## Compare implementations

Use `reference` for portable Kotlin and `host` for the selected host backend.
Benchmark setup prints the resolved backend; check it before using the result.
Suites compare complete implementations. They do not tune runtime size routing.

`level1` uses `kernels=builtin|host` instead. On the JVM, portable SIMD and
portable scalar are separate benchmark datasets: run once normally and once
with `-Pkoblas.noSimd=true`. The JVM benchmark task rewrites the saved result
arm as `reference-simd` or `reference-scalar` (and the level-1 arm as
`builtin-simd` or `builtin-scalar`). These are execution variants of the same
reference backend, not separate runtime providers. Setup output names the
resolved kernel variant; verify it before keeping a report.

Measure on the platform whose routing you are changing. JVM and native have
different Kotlin and FFI costs.

`tools/bench-curves.py` exports CSV and fits each arm. Use the comparison to
accept or reject an implementation and to rank further work by representative
workload impact. A small-size crossover is not a reason to add a dispatch gate.

## Priorities from existing runs

| Priority | Work | Evidence |
| --- | --- | --- |
| 1 | Reuse native sparse descriptors and factors | Copying CSC descriptors and factors dominates several sparse product and solve rows; retaining them changes the comparison materially. |
| 2 | Improve sparse factorization and repeated solves | KLU, UMFPACK, CHOLMOD, and SPQR produce sustained wins once the problems contain meaningful sparse work. |
| 3 | Improve portable kernels where they remain the implementation | Sparse triangular operations and unsupported product shapes have no host path to select, so portable improvements benefit every backend. |
| Later | Small dense and vector-call overhead | The measured differences are confined to tiny inputs and do not matter enough to justify runtime routing complexity. |

When two providers implement the same semantic role, use representative A/B
runs to set their registration priority. Do not turn individual curve crossings
into per-call policy.

## Verify the run

`BUILD SUCCESSFUL` does not prove JMH ran: a stale `/tmp/jmh.lock` can make it
exit zero. Mark the time before running, then find a newer report:

```bash
marker=$(mktemp)
./gradlew :koblas-bench:jvmLevel1Benchmark
find koblas-bench/build/reports/benchmarks/level1 -name '*.json' -newer "$marker"
```

No output means no measurement. Remove the stale lock and any stray `ForkedMain`
before retrying. Report directories may be reused with Gradle's configuration
cache, so copy results you need before the next run.

## Registration soak

```bash
./gradlew :koblas-bench:stressRegistration
./gradlew :koblas-bench:stressRegistration -Prounds=1000000 -Pthreads=8
```

This checks concurrent backend registration. It fails if a weaker offer wins.
