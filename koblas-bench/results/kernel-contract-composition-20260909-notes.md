# Kernel contract composition benchmark notes

`kernel-contract-composition-20260909.tar.gz` preserves eight raw JMH JSON reports used as a bounded
performance check for the vector/panel/packed responsibility split in PR #511. These are selected developer
runs, not the versioned contributor profile.

## Provenance and environment

The four baseline reports were produced from the clean checkout
`878807e575d56cb5f54e2b43ebf2795390a35fb3` (`origin/main` at measurement time). The four candidate reports
were produced from a dirty worktree based on that commit while the responsibility split was in progress. The
exact candidate tree hash and dirty diff were not captured in the reports. Its production arithmetic corresponds
to the implementation later committed through `54ed0db1a07563b4070d8acac81b71991ccb4add`; subsequent edits included
contract documentation, formatting, benchmark manifests, and tests. The candidate data therefore must not be
presented as a measurement of that final commit.

The runs used Linux x86-64 on a 12th Gen Intel Core i9-12900H with 20 online logical CPUs. `taskset -c 2-5`
pinned each Gradle/JMH process to CPUs 2–5. The JVM recorded in every JSON file is Eclipse Temurin 25.0.1+8,
with `jdk.incubator.vector` enabled. Each benchmark used one thread, one fork, three 500 ms warmups, and five
500 ms measurements. The built-in arm resolved to the JVM SIMD family with four double lanes. Other processes
were active on the shared host; the load was not captured alongside each run and no idle-machine or exclusive-CPU
claim is made.

## Commands

The end-to-end command was run twice for both the baseline and candidate, changing only `bench.reportsDir`.
The first pass in each pair also used `--rerun-tasks`:

```bash
taskset -c 2-5 ./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.reportsDir=/tmp/koblas-arch-perf/NAME \
  -Pbench.include='Level2Benchmark.gemv|Level2Benchmark.symv|Level3Benchmark.gemm|Level3Benchmark.gemmTransposedA|TrsmBenchmark.denseTrsm' \
  -Pbench.param.n=257 \
  -Pbench.param.denseArm=built-in \
  -Pbench.param.shape=128x256 \
  -Pbench.param.variant=right-lower-transposed
```

The direct packed command was likewise run twice per checkout, with `--rerun-tasks` on the first pass:

```bash
taskset -c 2-5 ./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.reportsDir=/tmp/koblas-arch-perf/NAME \
  -Pbench.include='ExplicitPackedKernelBenchmark.gemmTile|ExplicitPackedKernelBenchmark.trsmTile|ExplicitPackedKernelBenchmark.gemmTrsmTile' \
  -Pbench.param.depth=128 \
  -Pbench.param.edge=full \
  -Pbench.param.variant=lower-nonunit \
  -Pbench.param.kernels=built-in
```

`NAME` was `base-end-1`, `base-end-2`, `current-end-1`, `current-end-2`, `base-packed-1`,
`base-packed-2`, `current-packed-1`, or `current-packed-2` as applicable.

## Results

Average scores from the two passes are shown as `pass 1 / pass 2`; the JSON files retain raw samples,
99.9% confidence intervals, and JMH metadata.

| Benchmark | Baseline | Candidate | Unit |
| --- | ---: | ---: | --- |
| GEMV, order 257 | 6.043 / 9.163 | 7.120 / 7.876 | us/op |
| SYMV, order 257 | 9.486 / 10.595 | 9.083 / 8.409 | us/op |
| GEMM, order 257 | 1796.317 / 1402.018 | 1054.170 / 1111.131 | us/op |
| GEMM with transposed A, order 257 | 1419.313 / 1266.362 | 1078.835 / 1070.723 | us/op |
| right/lower/transposed TRSM, 128x256 | 414.501 / 378.836 | 301.477 / 301.629 | us/op |
| packed GEMM tile, depth 128 | 212.633 / 169.789 | 186.602 / 325.330 | ns/op |
| packed GEMM-TRSM tile, depth 128 | 285.567 / 255.573 | 288.174 / 630.694 | ns/op |
| packed TRSM tile | 3508.503 / 3055.869 | 3040.674 / 12103.519 | ns/op |

The end-to-end candidate ranges overlap or fall below the noisy baseline ranges. The second candidate packed
pass was contention dominated: its confidence intervals are extremely wide, including about 27.7 us for a
12.1 us packed TRSM estimate. The first candidate packed pass overlaps the baseline. These data guard against
an obvious structural regression but do not establish a performance improvement or a new crossover.

The direct packed benchmark setup printed `0 B/call` for GEMM, TRSM, and GEMM-TRSM in every observed baseline
and candidate run. Console logs were not retained and JMH's JSON has no allocation secondary metric, so the
archive cannot independently substantiate those observations. Post-change allocation behavior is separately
covered by the repository's JVM allocation tests and the full CI gate. No baseline/candidate allocation delta is
claimed from this archive.

The archive contains no OpenBLAS or oneMKL comparison, no Native timing, and no ARM or macOS timing. Numerical
correctness was validated separately through the scalar, bundled-C, SIMD, Linux Native, and cross-compilation
test matrix described in PR #511.

Expected SHA-256:
`74b186c6d0a26656288f9f89386f84d308a3034bff6bd927d58cf96d8231f3e6`.
