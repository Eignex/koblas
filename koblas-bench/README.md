# koblas-bench

Development benchmarks for every reviewed public numerical operation in koblas. This module is not published.

## Send a report

```bash
koblas-bench/report.sh jvm
koblas-bench/report.sh native
koblas-bench/report.sh jvm openblas
koblas-bench/report.sh jvm oneMkl
```

The command creates one archive containing two independently executed raw JSON passes, both benchmark logs,
metadata, and both coverage manifests. It rejects stale or byte-identical passes and requires exactly one fresh
result from each process. Metadata includes the UTC timestamp, commit and dirty status, OS, architecture, CPU
model/count, Gradle/JVM versions, target, command, affinity, allocation expectations, and resolved implementation
output. Inspect the archive before sending it: these details can identify your machine and checkout.

`report` is the complete built-in inventory. `openblas` is the dense OpenBLAS comparison and `oneMkl` is the
dense and sparse oneMKL comparison. Each external profile runs in separate benchmark processes so their global
symbols and thread controls cannot interfere. OpenBLAS and oneMKL are both forced to one thread. The bindings use
libraries installed on the development machine; absence fails an explicitly requested external profile instead
of falling back to koblas.

## Run benchmarks

```bash
./gradlew :koblas-bench:jvmFullBenchmark
```

For local A/B work:

```bash
./gradlew :koblas-bench:jvmSelectedBenchmark \\
  -Pbench.include='Level3Benchmark.gemm|Level3Benchmark.syrk' \\
  -Pbench.param.n=256 \\
  -Pbench.param.denseArm=built-in,openblas
```

### Arms

Dense parity uses `denseArm=built-in,openblas,onemkl`; retained sparse BLAS uses
`sparseArm=built-in,onemkl`. These arms construct the built-in implementation or open a benchmark-owned external
binding directly. They never use production discovery, and every setup asserts and reports its resolved identity.

The older `automatic` and `reference` backend parameters remain only for factorization suites during the later
consumer-coordinated removal phase. They are not external parity evidence. Kernel microbenchmarks retain `scalar`,
`c`, and `simd` pins, while `built-in` selects SIMD, then bundled C, then scalar without consulting the registry.

Benchmark-owned level-1 CBLAS calls cover dot, axpy, scale, norm, absolute sum, swap, and rotations for both dense
comparators. `sum` and fused squared distance have no CBLAS counterpart. Four-way dot is labeled as a composition
of four calls, never direct kernel parity. See `comparator-coverage.tsv` for the same direct/composition/missing
classification across dense and sparse operations.

`simd` is absent from every `@Param` list because Kotlin/Native has no such
provider and a benchmark configuration covers every target, so a full native
run would ask for one that cannot exist. Pass it explicitly on the JVM.

Every arm checks what it actually resolved to and fails when that is not what
the arm names, so a run cannot quietly credit an implementation that never
executed. Each install prints one `resolved: arm=...` line naming the arm and
the halves behind it, and `report.sh` collects those lines into the report.

Sparse one-shot rows include oneMKL CSC conversion and destruction. Prepared rows retain the inspector-executor
handle across invocations. Fresh sparse results and packing are reported with workload-dependent allocation
expectations; allocation-free JVM kernels are probed in every fork and invalidate it if managed allocation exceeds
the near-zero allowance.

## Troubleshooting and maintenance

A successful Gradle task without fresh JSON can mean a stale JMH lock. `report.sh` checks for JSON newer than its marker and reports a detected lock; confirm no benchmark is running before removing a stale lock.

When adding or changing a public numerical API, update `public-numerical-api.tsv` and `benchmark-coverage.tsv`, then
run `./gradlew :koblas-bench:checkBenchmarkCoverage`. The inventory names each reviewed public operation and maps it
to a benchmarked manifest row or an explicitly justified exclusion. The check fails for a public operation absent
from the inventory, an inventory entry without a manifest row, an unlisted benchmark method, or a manifest method
that no longer exists.
