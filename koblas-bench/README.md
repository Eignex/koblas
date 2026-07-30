# koblas-bench

JMH benchmarks for koblas, via kotlinx-benchmark. Development only: this module
is not published and nothing depends on it.

## Running

```bash
./gradlew :koblas-bench:jvmLevel3Benchmark   # one suite
./gradlew :koblas-bench:jvmBenchmark         # every suite
```

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

`level1` has no such parameter: those kernels sit below the seam. Run it twice,
once as-is and once with `-Pkoblas.noSimd=true` to withhold the incubator
vector module, to compare SIMD against scalar.

Fixtures live in `BenchmarkFixtures.kt` and derive from one seed, so a re-run
measures the same operands rather than similar ones.
