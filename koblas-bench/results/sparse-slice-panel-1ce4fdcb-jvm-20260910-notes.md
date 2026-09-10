# Sparse slice and panel kernel benchmark notes

`sparse-slice-panel-1ce4fdcb-jvm-20260910.tar.gz` preserves four raw JMH JSON reports used as a bounded
performance check for the Phase 4 sparse numerical-leaf extraction. These are selected developer runs, not the
versioned contributor profile.

## Provenance and environment

The two baseline reports were produced from the clean detached checkout
`8f3b84829c433b444a098cbdb2a37b862b3cadd9` (`origin/main` immediately after PR #516). The first candidate
report was produced from a dirty worktree based on that commit; its complete source and test content is exactly
the tree later committed as `1ce4fdcbed25a6a55593e5b12d625922793898f1`. The second candidate report was
produced from that clean commit. No production source changed between the two candidate reports.

The runs used Linux 6.17 x86-64 on a 12th Gen Intel Core i9-12900H with 20 online logical CPUs. Gradle 9.7.1
used a Homebrew OpenJDK 22.0.2 launcher; JMH forked Eclipse Temurin 25.0.1+8-LTS from
`/home/rasmus/.gradle/jdks/eclipse_adoptium-25-amd64-linux.2/bin/java`. Each benchmark used one thread, one
fork, three 500 ms warmups, and five 500 ms measurements. The built-in sparse arm resolved to
`built-in/built-in/simd-sparse`. The shared host had no CPU affinity or reservation, so cross-pass variability is
part of the evidence and no idle-machine claim is made.

## Command

The command was run twice in the baseline checkout and twice in the candidate checkout, changing only
`bench.reportsDir`. `--rerun-tasks` was used for the first clean candidate pass and when needed to force a fresh
run.

```bash
./gradlew :koblas-bench:jvmSelectedBenchmark --rerun-tasks \
  -Pbench.reportsDir=/tmp/koblas-phase4-evidence/NAME \
  -Pbench.include='SparseLevel1ComparisonBenchmark.sparseDotDense|SparseLevel1ComparisonBenchmark.sparseAxpy|SparseCompletionBenchmark.symv|SparseCompletionBenchmark.symm|SparseProductHostBenchmark.gemv|SparseProductHostBenchmark.gemm|SparseProductHostBenchmark.trmv|SparseProductHostBenchmark.trsv|SparseProductHostBenchmark.trmm|SparseProductHostBenchmark.trsm' \
  -Pbench.param.len=4096 \
  -Pbench.param.n=256 \
  -Pbench.param.density=0.01 \
  -Pbench.param.sparseArm=built-in \
  -Pbench.param.lower=true \
  -Pbench.param.side=left \
  -Pbench.param.productShape=regular
```

`NAME` was `baseline-1`, `baseline-2`, `candidate-dirty-2`, or `candidate-clean-1`.

## Results

Average scores are shown as `pass 1 / pass 2`; the JSON files retain raw samples, 99.9% confidence intervals,
and JMH metadata.

| Benchmark | Baseline | Candidate | Unit |
| --- | ---: | ---: | --- |
| SYMM | 9.586 / 16.032 | 7.954 / 7.862 | us/op |
| SYMV | 0.949 / 1.368 | 1.018 / 1.042 | us/op |
| sparse AXPY, length 4096 | 22.536 / 37.085 | 24.202 / 24.205 | ns/op |
| sparse-dense dot, length 4096 | 18.414 / 29.411 | 19.206 / 19.988 | ns/op |
| GEMM | 7.671 / 8.929 | 6.677 / 6.807 | us/op |
| GEMV | 0.854 / 1.712 | 0.892 / 0.899 | us/op |
| TRMM | 5.630 / 10.883 | 10.191 / 9.971 | us/op |
| TRMV | 1.254 / 2.203 | 1.212 / 1.261 | us/op |
| TRSM | 12.216 / 11.844 | 9.877 / 9.725 | us/op |
| TRSV | 3.430 / 6.418 | 1.732 / 1.862 | us/op |

Every candidate score lies within or below the two-pass baseline range. The especially wide baseline spread for
TRMM, TRSV, SYMM, and the Level-1 rows makes a speedup claim inappropriate. These data meet the Phase 4
best-effort parity objective by ruling out an obvious structural regression at the selected call sites; they do
not establish a new crossover or a performance improvement.

The benchmark sources and runner plumbing were not changed. Allocation-sensitive evidence belongs to unit tests:
the JVM allocation suite covers sparse Level-1 dot, transposed sparse GEMV, sparse SYMV, sparse GEMM/SYMM, and
triangular paths after warmup. The archive contains no independent allocation metric and makes no allocation
delta claim.

The archive contains no OpenBLAS or oneMKL comparison, no Native timing, and no ARM or macOS timing. Numerical
correctness was validated separately through the scalar, bundled-C, JVM SIMD, runnable Linux x64 Native, and
cross-compilation matrix recorded in the pull request.

Expected SHA-256:
`56d2a242cc7d46c1c0e8405a549546563ca6ce53b75f4b5fd4fed1037487ce04`.
