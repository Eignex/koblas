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
| `blockSolve` | block versus per-column solves |
| `sparse` | sparse factorization, solve, and `gemv` |

## Compare implementations

Use `reference` for portable Kotlin and `forced` for the host backend with its
thresholds disabled. `auto` uses the shipped routing. Benchmark setup prints the
resolved backend; check it before using the result.

`level1` uses `kernels=builtin|host` instead. Run it with and without
`-Pkoblas.noSimd=true` when comparing SIMD and scalar Kotlin.

Measure on the platform whose routing you are changing. JVM and native have
different Kotlin and FFI costs.

`tools/bench-curves.py` exports CSV and fits each arm. A crossover requires the
curves to cross; different intercepts alone do not establish one.

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
