# Dense numerical leaf extraction benchmark notes

`dense-leaf-extraction-20260909.tar.gz` contains four raw JMH JSON reports from a bounded before/after
check of the phase-2 dense numerical leaf extraction. These are selected developer runs, not the versioned
contributor profile and not evidence of a new performance crossover.

## Provenance and environment

The two baseline passes used a clean detached worktree at
`39e49b390144cb99401f77ab2e252364ba1c7511`. The two candidate passes used the clean commit
`aaf5fcbb3ded60e1a7538c2ba620761d3c0845ee`. The archive retains both raw passes and SHA-256 checksums.

The runs used Linux x86-64 on a 12th Gen Intel Core i9-12900H with 20 online logical CPUs. `taskset -c 2-5`
pinned each Gradle/JMH process to CPUs 2–5; this affinity does not reserve them. Other work was allowed on the
shared host, so cross-pass movement and wide confidence intervals are treated as uncertainty. The measured JVM
was Eclipse Temurin 25.0.1+8-LTS with `jdk.incubator.vector`; the built-in arm resolved to immutable JVM SIMD
with four double lanes. Every case used one thread, one fork, three 500 ms warmups and five 500 ms measurements.

## Command

The command was run twice in each clean checkout, changing only `bench.reportsDir`; the first pass in each pair
also used `--rerun-tasks`.

```bash
taskset -c 2-5 ./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.reportsDir=/tmp/koblas-phase2-results/NAME \
  -Pbench.include='Level2Benchmark.gemv|Level2Benchmark.gemvTransposed|Level2Benchmark.symv|Level2Benchmark.symvUpper|Level2Benchmark.ger|Level2Benchmark.syr|Level2Benchmark.syr2|GemmtBenchmark.gemmt|TrsmBenchmark.denseTrsm' \
  -Pbench.param.n=257 \
  -Pbench.param.denseArm=built-in \
  -Pbench.param.productShape=129x257 \
  -Pbench.param.transposeA=true \
  -Pbench.param.transposeB=false \
  -Pbench.param.lower=true \
  -Pbench.param.shape=128x256 \
  -Pbench.param.variant=right-lower-transposed
```

## Results

Scores are pass 1 / pass 2 in microseconds per operation. The raw reports retain samples and 99.9% confidence
intervals.

| Benchmark | Baseline | Candidate |
| --- | ---: | ---: |
| GEMV | 15.769 / 8.192 | 8.029 / 6.458 |
| transposed GEMV | 6.468 / 8.010 | 8.261 / 6.795 |
| lower SYMV | 8.323 / 9.838 | 9.686 / 7.853 |
| upper SYMV | 9.066 / 10.334 | 9.647 / 8.858 |
| GER | 10.079 / 12.079 | 11.173 / 9.833 |
| SYR | 5.670 / 6.667 | 6.571 / 5.741 |
| SYR2 | 9.005 / 12.921 | 10.856 / 12.994 |
| transposed-A lower GEMMT | 412.055 / 338.792 | 311.999 / 298.494 |
| right/lower/transposed TRSM | 377.561 / 337.330 | 311.326 / 374.841 |

The before/after ranges overlap or move in both directions on the busy host. The extraction therefore shows no
material call-boundary regression and makes no speedup claim. The phase briefly introduced 960 B/call in the
right-side ordinary TRSM allocation test by storing Kotlin progressions as values; direct ordered indices removed
that regression before the measured candidate commit. Repeated repository allocation probes at the candidate
commit report zero managed bytes per warmed right-side solve call and zero for the existing small dense Level-2
probes. JMH console logs were not retained, so allocation evidence is supplied by the checked tests rather than
the archive.

No OpenBLAS, oneMKL, ARM, macOS, or Native timing was collected. Correctness and platform compilation are covered
by the separately recorded validation matrix. Archive SHA-256:
`9379c05859d8df37429ee1d9908e51c178f109c77ff0772425fa9384a905814c`.
