# koblas-bench

Benchmarks for koblas, via kotlinx-benchmark: JMH on the JVM, kotlinx-benchmark's
own harness on native. Development only: this module is not published and nothing
depends on it.

## Running

```bash
./gradlew :koblas-bench:jvmLevel3Benchmark       # one suite, JVM
./gradlew :koblas-bench:linuxX64Level3Benchmark  # the same suite, native
./gradlew :koblas-bench:jvmBenchmark             # every suite
```

Prefix a suite with the target: `jvm`, `linuxX64` or `macosArm64`. Only the host's
own target can run, so macOS numbers come from a macOS checkout. Native binaries
link in release mode, so the portable side is optimized Kotlin.

One suite per benchmark class, named after it:

| Suite        | Measures                                                 |
| ------------ | -------------------------------------------------------- |
| `level1`     | `dot`, `axpy` across vector lengths                      |
| `level2`     | `gemv`, `symv` from small to bandwidth-bound sizes        |
| `level3`     | `gemm`, `syrk` in both triangle modes                    |
| `solve`      | LU factorization, LU and LDL vector solves               |
| `blockSolve` | block against per-column solves as `nrhs` grows          |
| `sparse`     | sparse LU factorize plus solve, sparse `gemv`            |

Warmup and iteration settings are shared by every suite in `build.gradle.kts`,
so two runs are comparable. Results land in
`build/reports/benchmarks/<suite>/<timestamp>/jvm.json`.

## Reading the output

Classes that cross the backend seam take a `backend` parameter: `auto` resolves
whatever is on the classpath (OpenBLAS here) and `reference` forces the portable
kernels, so one run gives both sides of the comparison. Every `@Setup` prints
the resolved backend — check it before trusting a number, because a missing
native library changes what was measured without failing anything.

`level1` has no such parameter: those kernels reach a backend through VectorKernels
rather than through the `backend` the other suites switch. Run it twice,
once as-is and once with `-Pkoblas.noSimd=true` to withhold the incubator
vector module, to compare SIMD against scalar.

**Do not carry a conclusion from one platform to the other.** The JVM's portable
kernels are SIMD and its FFI is expensive, so it keeps level 2 and the vector
solves in Kotlin. Native's portable kernels are scalar and its FFI is cheap, so
the host BLAS wins those same routines by 2x to 15x. Whichever platform a routing
decision is about, measure it there.

Fixtures live in `BenchmarkFixtures.kt` and derive from one seed, so a re-run
measures the same operands rather than similar ones.

## Check that it ran

A benchmark task reports `BUILD SUCCESSFUL` even when JMH refused to start — a
killed run leaves `/tmp/jmh.lock` behind, and the next invocation prints
`Failure: Unable to acquire the JMH lock` and still exits zero. The report
directory is the only reliable evidence:

```bash
ls -t koblas-bench/build/reports/benchmarks/<suite>/
```

If the newest timestamp predates your run, nothing was measured and any numbers
you are reading are from a previous one. Remove the lock and check for a stray
`ForkedMain` process before re-running.

## Soaks

Not every regression shows up as a time. `stressRegistration` looks for a wrong
answer instead:

```bash
./gradlew :koblas-bench:stressRegistration
./gradlew :koblas-bench:stressRegistration -Prounds=1000000 -Pthreads=8
```

It has several threads offer a band of backend priorities at once and checks the
strongest one ends up holding the half. `Seam.register` takes its offer with a
compare-and-set for exactly this reason, and the task fails with a non-zero exit
if a weaker offer ever wins.

It lives here rather than in the test suite because catching a regression needs
the offers to collide, which is luck per round. Measured against the registration
as it was before the compare-and-set, a weaker offer won about once every ten
thousand rounds, so the default is 200,000 rounds and takes about eight seconds.
A run prints `contended=` alongside `rounds=`; if those are not equal the threads
were not really racing and the run proves less than it looks.
