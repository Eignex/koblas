# Retained BLAS gap evidence

These selected JVM measurements ran on an Intel Core i9-12900H with Eclipse Adoptium JDK 25.0.1+8-LTS.
The host was shared and no reservation or idle period was requested. Each process was pinned to logical CPUs
`2-5`; OpenBLAS was fixed to one thread. JMH's advisory global lock was ignored where another process held it,
and the resulting uncertainty is retained rather than filtered.

Preflight resolved the built-in implementation and OpenBLAS 0.3.30 independently. `libmkl_rt` was unavailable,
so no oneMKL timing or substitute comparator is included. The oneMKL binding still compiles on JVM, and the
macOS ARM64 and Linux ARM64 source sets compile as part of the project gate.

## Source provenance

The four original files were measured with Git HEAD
`14d758879d42b546371d3f7613babb0096ade8ee` and a dirty working tree. The relevant changes were subsequently
committed as `8117c9c6`, `eefad2b7`, and `d66fa8d4`; no dirty-tree patch hash was captured. These files are retained
as pre-fix evidence for the originally reviewed #507 head and are not presented as clean-commit measurements.

The four `*-fixed-*` files were measured from clean commit
`a11c6d70c624719598fe2c692d404b50d6809f1a`, the corrected #507 head. They replace performance claims for the
changed GEMMT eligibility scan and sparse SYRK adjacency traversal without relabeling the earlier results.

## Commands

Capability detection used:

```bash
koblas-bench/report.sh preflight jvm
```

The original GEMMT passes used the following command, first with `denseArm=built-in,openblas` and then with
`denseArm=openblas,built-in`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 2-5 ./gradlew \
  :koblas-bench:jvmSelectedBenchmark \
  -Pbench.include='GemmtBenchmark.gemmt' \
  -Pbench.param.productShape=129x257 \
  -Pbench.param.transposeA=false -Pbench.param.transposeB=false \
  -Pbench.param.lower=true -Pbench.param.denseArm=built-in,openblas
```

The original sparse passes used this command twice:

```bash
JAVA_TOOL_OPTIONS=-Djmh.ignoreLock=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 2-5 ./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.include='SparseCompletionBenchmark.*' \
  -Pbench.param.n=129 -Pbench.param.density=0.01 \
  -Pbench.param.lower=true -Pbench.param.sparseArm=built-in
```

The corrected GEMMT passes used the GEMMT command above with `JAVA_TOOL_OPTIONS=-Djmh.ignoreLock=true`, again
in both arm orders. The corrected sparse SYRK passes used this command twice:

```bash
JAVA_TOOL_OPTIONS=-Djmh.ignoreLock=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  taskset -c 2-5 ./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.include='SparseCompletionBenchmark.syrkDense|SparseCompletionBenchmark.syrkSparse' \
  -Pbench.param.n=129 -Pbench.param.density=0.01 \
  -Pbench.param.lower=true -Pbench.param.sparseArm=built-in
```

## Results

`gemmt-pass-1.json` and `gemmt-pass-2.json` contain the original reversed-order built-in/OpenBLAS comparisons
for a lower `129x129` result with depth 257. The built-in measurements were
`244.595 +/- 10.815 us/op` and `260.609 +/- 17.351 us/op`; OpenBLAS measured
`262.800 +/- 9.710 us/op` and `256.583 +/- 9.624 us/op`.

`sparse-pass-1.json` and `sparse-pass-2.json` each contain all nine original sparse benchmark rows at order 129
and density 0.01. Sparse-result SYRK measured `25.444 +/- 11.421 us/op` and
`27.035 +/- 3.257 us/op`. The first value is the exact rounded form of the archived score
`25.443912547654076` with error `11.420868218839418`; the previously published `29.375 +/- 14.641` value came
from an earlier run that was not copied into the archive.

`gemmt-fixed-pass-1.json` and `gemmt-fixed-pass-2.json` measure the corrected clean commit. Built-in GEMMT measured
`266.900 +/- 33.548 us/op` and `308.677 +/- 65.636 us/op`; OpenBLAS measured
`248.915 +/- 28.633 us/op` and `266.230 +/- 64.580 us/op`. The shared-host confidence intervals overlap.

`syrk-fixed-pass-1.json` and `syrk-fixed-pass-2.json` measure the adjacency traversal at the same sparse shape.
Dense-result SYRK measured `3.907 +/- 1.306 us/op` and `3.702 +/- 0.662 us/op`; sparse-result SYRK measured
`3.611 +/- 1.140 us/op` and `4.221 +/- 0.696 us/op`. The old traversal's structural search cost is absent.

Before the corrected measurements, `./gradlew :koblas:check :koblas-bench:check lintDocs` and the complete JVM
suite with `-Pkoblas.noSimd=true` passed. The gate includes Linux x64 tests, Linux ARM64 and macOS ARM64
compilation, ABI checks, documentation lint, and the 97-method benchmark coverage check. Focused tests cover the
GEMMT arithmetic-order counterexamples, diagonal and hypersparse SYRK in both orientations, and warmed allocation.
