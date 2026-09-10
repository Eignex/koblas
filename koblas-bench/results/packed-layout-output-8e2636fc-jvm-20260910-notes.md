# Packed layout and output leaves

## Provenance

- Architecture phase: 3, packed layout and output leaves.
- Exact merged baseline: `533f4d28c1989d4249735d90c5f88d8db824e9d1` (`main` after PRs #512--#514).
- Exact final measured code: `8e2636fca8ec11321e7804e5bc4e1137d6023ff8`.
- Current PR head after measurement: `0f55a9cad201abe92582a46239badf6615dc8eba`. Its final source correction changes
  unit-diagonal scaled packing from `alpha` to `alpha * 1.0` to preserve the previous floating-point evaluation
  order; none of the archived timing rows measure that correction.
- Baseline state: detached, clean temporary worktree.
- Final state: clean branch worktree; `git status --porcelain=v1` was empty before the `clean-1` and `clean-2`
  series.
- Interleaved candidate: `04bce0eb35045a7adab0c0ec5a40cb264c186942`, with the same executable source as
  `8e2636fc` (only formatter whitespace differs). Its tracked source and index were clean, while one preliminary
  untracked evidence archive was present. Those `final-1` and `final-2` rows ran before the clean baseline and
  final-clean repetitions and are retained to expose host variation, not substituted for the clean comparison.
- Archive SHA-256: `d356d94016baffe44c54f1b744faf0d68beebcd2a81357a498a2b22275455503`.

The archive contains 24 raw JMH JSON files: two clean baseline series, two interleaved candidate series and two
final clean candidate series, each split into panels, products, rank updates and triangular operations.

## Environment

- Linux `6.17.0-41-generic`, x86-64.
- 12th Gen Intel Core i9-12900H, 20 logical CPUs.
- JMH fork JVM recorded in every archived result: OpenJDK 25.0.1, VM `25.0.1+8-LTS`.
- Gradle launcher and daemon JVM: OpenJDK 22.0.2. Gradle 9.7.1 reports embedded Kotlin 2.4.0; the build selects
  Kotlin Gradle plugins 2.4.10.
- Built-in immutable SIMD engine, reported as `built-in/built-in/simd(4 lanes)`.
- One JMH fork, three 500 ms warmups and five 500 ms measurements per case.
- No CPU affinity, reservation or peer pause. The shared host was visibly busy and became more contended during
  the final clean series. No external comparator was requested; every benchmark used one calling thread.
- ARM and macOS timings were not measured.

## Commands

Each command below was run twice on the clean merged base (`base-1`, `base-2`), twice on the interleaved candidate
(`final-1`, `final-2`) and twice on the clean final candidate (`clean-1`, `clean-2`), changing only the report path.

```bash
./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.reportsDir=koblas-bench/results/packed-layout-phase3-20260910/<series>/panels \
  -Pbench.include='PackedPanelBenchmark.(packLeft|packRight|writeLeft|writeRight)' \
  -Pbench.param.depth=31 -Pbench.param.edge=full,partial

./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.reportsDir=koblas-bench/results/packed-layout-phase3-20260910/<series>/products \
  -Pbench.include='Level3Benchmark.(gemm|gemmTransposedA|symm|symmRight)' \
  -Pbench.param.n=129 -Pbench.param.denseArm=built-in

./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.reportsDir=koblas-bench/results/packed-layout-phase3-20260910/<series>/rank \
  -Pbench.include='(SyrkBenchmark.syrk|Syr2kBenchmark.syr2k)' \
  -Pbench.param.rankShape=129x257 -Pbench.param.transpose=false,true \
  -Pbench.param.lower=true,false -Pbench.param.denseArm=built-in

./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.reportsDir=koblas-bench/results/packed-layout-phase3-20260910/<series>/triangular \
  -Pbench.include='(TrmmBenchmark.denseTrmm|TrsmBenchmark.denseTrsm)' \
  -Pbench.param.shape=129x31 \
  -Pbench.param.variant=left-lower,right-upper-transposed \
  -Pbench.param.denseArm=built-in
```

## Results

The clean score ranges below deliberately retain contention rather than selecting a favorable pass. Units are
ns/op for panel operations and us/op otherwise.

| operation family | merged base range | final clean range |
| --- | ---: | ---: |
| left packing, full/partial | 190--243 | 295--466 |
| right packing, full/partial | 126--187 | 257--483 |
| left writeback, full/partial | 150--191 | 117--216 |
| right writeback, full/partial | 111--171 | 118--250 |
| GEMM/transposed GEMM | 136--179 | 198--427 |
| left/right SYMM | 145--260 | 207--247 |
| SYRK, all orientation/triangle cases | 193--424 | 379--561 |
| SYR2K, all orientation/triangle cases | 428--773 | 738--1426 |
| left/right TRMM | 55--103 | 106--205 |
| left/right TRSM | 61--116 | 115--174 |

These clean ranges do not establish 1.1x parity. The semantically identical interleaved candidate series is much
less contended: GEMM/SYMM ranges overlap the base; all four SYR2K minima and all four SYRK minima are within
or below the base ranges; TRMM/TRSM move in both directions; and packing/writeback is at parity or faster. The raw
confidence intervals are often wide in every series. The combined evidence therefore supports only that the
dedicated loop shapes removed the initial tiny-array-copy regression; it does not support a performance gain or a
stable slowdown claim on this shared host.

Every successful setup probe for full and partial `packLeft`, `packRight`, `writeLeft` and `writeRight` reported
`0 B/call` in all six series. No per-tile wrapper, callback or progression allocation was introduced.

## Verification

```bash
./gradlew :koblas:check :koblas-bench:check lintDocs
./gradlew :koblas:jvmTest -Pkoblas.noSimd=true
```

The full gate passed, including JVM tests, Linux x64 Native tests, Linux arm64 and macOS arm64 compilation, ABI,
coverage, static analysis and documentation checks. The no-SIMD JVM suite passed through the bundled C family;
the common packed tests also exercise the explicit portable scalar implementation.
