# Sparse workspace comparator evidence

The retained archive contains two independent JVM passes and one Linux x86-64 Native pass from the
`SparseWorkspaceOneMklComparisonBenchmark`, `SparseWorkspaceGrowingComparisonBenchmark`, and
`SparseWorkspaceBaselineComparisonBenchmark` developer suites. All used `BENCH_SEED=20260730`, count 64,
dimension 513, 50% first touches, exact-zero compaction, support 512, four-entry short updates, one benchmark thread,
three 500 ms warmups, five 500 ms measurements, and one fork. JVM passes were pinned independently with `taskset`
to logical CPUs 4 and 6; Native used CPU 3. Other repository builds and benchmarks were allowed to run, so the raw
confidence intervals and cross-pass variation are authoritative.

Measurements used sparse comparison implementation commit `97aba16a2bdcf421465c69c2c38c0f8b3c3d2b3f`, stacked
on contributor-runner commit `6d1644270716f759cb77958cfb0637f3ca1e6f8c` (PR #503).

The command shape was:

```bash
taskset -c CPU ./gradlew :koblas-bench:jvmSelectedBenchmark \
  -Pbench.include='SparseWorkspaceOneMklComparisonBenchmark.*|SparseWorkspaceGrowingComparisonBenchmark.manyShortScattersEquivalent|SparseWorkspaceBaselineComparisonBenchmark.*' \
  -Pbench.param.count=64 -Pbench.param.firstTouchPercent=50 \
  -Pbench.param.compactExactZeros=true -Pbench.param.sparseArm=built-in \
  -Pbench.param.supportSize=512 -Pbench.param.scatterSize=4 \
  -Pbench.param.baselineArm=built-in,baseline
```

The corresponding Native command used `:koblas-bench:linuxX64SelectedBenchmark` and omitted the unsupported
oneMKL arm. Every warmed JVM allocation probe reported 0 B/call. Native managed-allocation measurement is
unavailable and is labeled as such.

At 64 entries, both JVM passes put the independent active-maximum, candidate-filtering, and checked-scatter
baselines near the public implementations within their observed uncertainty. The first complete post-correctness
pass measured approximately 194/196 ns for built-in/baseline active maximum, 155/154 ns for candidates, and
441/433 ns for checked scatter. The Native pass measured 222/213 ns, 250/229 ns, and 1116/777 ns respectively.
The Native checked-baseline gap is retained as an implementation comparison; no production tuning was performed.

`libmkl_rt` was not installed on this host. Consequently no oneMKL timing or allocation claim is present, and the
archive does not substitute another implementation. The composed oneMKL path is correctness-gated against the
public contract when the library is available; an independent indexed primitive supplies the same gate on hosts
without oneMKL. Tests cover nonzero and empty slices, stale accumulator entries, first-touch order, mixed and repeated
overlap, cancellations and signed zeroes, compaction, nonfinite gathers, active masks, thresholds, alpha-zero and
underflow diagnostics, capacity failure before mutation, and complete accumulator/mark/support/output state.

The contributor runner profile `koblas-hardware-standard` v1 stays immutable. These are developer selected-command
results; catalog inclusion is deferred to a deliberate later profile version.
