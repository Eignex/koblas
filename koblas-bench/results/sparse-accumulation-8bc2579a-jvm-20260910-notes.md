# Sparse accumulation ownership evidence

This archive retains six raw JVM JMH JSON files measured for Phase 5 of the architecture sweep. The clean
baseline is merge commit `c0e6945867917f8ce2e9e8863f97f5314981d860` (merged PR #517). The clean candidate is
`8bc2579a079fb825b8e9760d29ac8789669f3ff5`, including the sparse accumulation extraction and the follow-up
inline boundary for hot slice and column leaves. Both worktrees were clean for every retained run.

The machine was a shared Intel Core i9-12900H running Linux 7.0.0-31-generic. No CPU affinity or host
reservation was requested. The JVM was intentionally JDK 25: Eclipse Temurin 25.0.1+8-LTS at
`/home/rasmus/.gradle/jdks/eclipse_adoptium-25-amd64-linux.2/bin/java`. Every run used one thread, one fork,
three 500 ms warmups, and five 500 ms measurements. The built-in sparse arm resolved to
`built-in/built-in/simd-sparse` for the full-operation runs; workspace primitives resolved to `portable`.

The full-operation command was:

```bash
./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.include='SparseCompletionBenchmark.(sparseProductScaledTransposed|denseProduct|syrkDense|syrkSparse|addScaled)|SparseBenchmark.(sparseSyr|sparseSyr2)' \
  -Pbench.param.n=256 -Pbench.param.density=0.1 -Pbench.param.lower=true \
  -Pbench.param.side=left -Pbench.param.sparseArm=built-in \
  -Pbench.reportsDir=REPORT_DIRECTORY
```

The shorter repeated product/addition command used the same parameters and:

```bash
-Pbench.include='SparseCompletionBenchmark.(sparseProductScaledTransposed|denseProduct|addScaled)'
```

The workspace command was:

```bash
./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.include='SparseWorkspaceBenchmark.(scatterFirstTouch|scatterExistingSupport|gatherTouched|gatherClearTouched|activeMaxAbs|filterCandidatePositions)' \
  -Pbench.param.count=512 -Pbench.reportsDir=REPORT_DIRECTORY
```

The first full baseline/candidate pass was strongly affected by changing host contention. Baseline/candidate
scores in microseconds were 20.136/33.812 for sparse SYR, 157.872/257.159 for sparse SYR2,
11.395/30.449 for scaled addition, 75.321/172.268 for dense product, 468.478/633.267 for scaled transposed
sparse product, 678.734/469.277 for dense SYRK, and 892.405/877.999 for sparse SYRK. Several candidate
confidence intervals were wider than their scores.

The immediately repeated product/addition pass measured baseline/candidate scores of 19.788/17.416 us for
scaled addition, 158.584/135.049 us for dense product, and 515.204/432.725 us for scaled transposed sparse
product. These contemporaneous results do not show an obvious structural regression, but the broad intervals
and reversal from the first pass do not support a speedup claim.

The workspace baseline/candidate scores in nanoseconds were 1419.931/1297.781 for active maximum,
952.777/1214.643 for candidate filtering, 787.739/995.477 for gather-and-clear, 339.744/450.248 for gather,
1027.789/1099.305 for existing-support scatter, and 2171.012/1306.475 for first-touch scatter. Variability was
again substantial, especially for first-touch baseline and candidate filtering.

Benchmark setup printed warmed `0 B/call` observations, but they are not treated as allocation evidence here.
Allocation behavior is gated by the JVM unit tests, including reusable sparse-to-dense destinations and
exception-safe return of every borrowed SYRK scratch buffer. Correctness was separately validated through the
default JVM SIMD suite, `-Pkoblas.noSimd=true` bundled-C JVM suite, Kotlin/Native execution and compilation,
and HFactor checks. No ARM, macOS, OpenBLAS, or oneMKL timing claim is made.

Expected archive SHA-256:
`dc42f9ae2dba5c2f6b0c367603fe53b5cc2dd5bea951ba69f5eebaa394fb8789`.
