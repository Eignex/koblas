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

`level1` takes a `kernels` parameter instead, since those kernels sit below the
`backend` seam the other suites switch: `builtin` leaves the compiled-in kernels
in place and `host` installs the host CBLAS ones. The host arm installs past the
level-1 gate on purpose, because that gate is what such a run exists to measure.
On a JVM with the Vector API kernels it is `Int.MAX_VALUE`, so an arm left to the
shipped default routes every length back to the compiled-in kernels and times
them twice. Run it twice as well, once as-is and once with
`-Pkoblas.noSimd=true` to withhold the incubator vector module, to compare SIMD
against scalar.

`forced` is a third value the level-2 and level-3 `backend` parameter takes. It
installs the host halves with their level-2, level-3 and factorization gates at
zero, for the same reason: a curve measured under the shipped gate shows one side
below the threshold and the other above it, which is two half-curves rather than
the two full ones a crossover is read off. `reference` and `forced` give those,
and `auto` keeps what the shipped gates actually do.

`tools/bench-curves.py` turns a report into per-arm CSV plus a least-squares fit
of `ns = intercept + slope * size`. Read the slopes first: a gate exists only
where two curves cross, so two arms with the same slope have no crossover at any
threshold, however far their intercepts differ. That is a result, not a failed
measurement.

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
`Failure: Unable to acquire the JMH lock` and still exits zero. A freshly
written report is the only evidence that anything ran, so key on the mtime of
the report json against a marker dropped just before the run:

```bash
marker=$(mktemp)
./gradlew :koblas-bench:jvmLevel1Benchmark
find koblas-bench/build/reports/benchmarks/<suite> -name '*.json' -newer "$marker"
```

Nothing newer than the marker means nothing was measured, and any numbers you
are reading are from a previous run. Remove the lock and check for a stray
`ForkedMain` process before re-running.

Do not compare directory names instead. The timestamped directory is named
during configuration, so a reused configuration cache entry (`Configuration
cache entry reused`) hands the run an earlier one's name and it writes *into*
that directory, overwriting the report already there. A name comparison then
reads a real run as "nothing measured" while the previous run's numbers are
silently destroyed. To detect that overwrite, snapshot the names before the run
and check whether the fresh report landed in one that already existed:

```bash
before=$(ls koblas-bench/build/reports/benchmarks/<suite>)
# ... run, then locate the fresh report as above ...
printf '%s\n' "$before" | grep -qxF "$(basename "$(dirname "$report")")" \
  && echo 'WARNING: overwrote an earlier report'
```

Extract anything you need from a report before starting the next run; the copy
in `build/` is not durable.

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
