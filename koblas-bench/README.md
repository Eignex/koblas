# koblas-bench

Development benchmarks for every reviewed public numerical operation in koblas. This module is not published.

## Send a report

```bash
koblas-bench/report.sh jvm
koblas-bench/report.sh native
```

The command creates one archive containing fresh raw JSON, the benchmark log, metadata, and the coverage manifest. Metadata includes the UTC timestamp, commit and dirty status, OS, architecture, CPU model/count, Gradle/JVM versions, target, command, affinity, and resolved backend output. Inspect the archive before sending it: these details can identify your machine and checkout.

## Run benchmarks

```bash
./gradlew :koblas-bench:jvmFullBenchmark
```

For local A/B work:

```bash
./gradlew :koblas-bench:jvmSelectedBenchmark \\
  -Pbench.include='Level3Benchmark.gemm|Level3Benchmark.syrk' \\
  -Pbench.param.n=256 \\
  -Pbench.param.backend=reference,host
```

### Arms

`automatic` measures normal production discovery. On a machine with a host
library installed, discovery selects it, so `automatic` **is** the host backend
and is never the portable arm. Reading it as the portable side of a comparison
puts the same library on both sides and produces a table that means nothing.

The arms that pin an implementation are `reference` for the portable matrix
routines, `scalar`, `c` and `simd` for a single built-in kernel provider, and
`host` for the host binding. A comparison of portable against host has to pin
both sides: `-Pbench.param.backend=reference,host` for matrix routines and
`-Pbench.param.kernels=simd,host` for kernels.

`simd` is absent from every `@Param` list because Kotlin/Native has no such
provider and a benchmark configuration covers every target, so a full native
run would ask for one that cannot exist. Pass it explicitly on the JVM.

Every arm checks what it actually resolved to and fails when that is not what
the arm names, so a run cannot quietly credit an implementation that never
executed. Each install prints one `resolved: arm=...` line naming the arm and
the halves behind it, and `report.sh` collects those lines into the report.

The report profile compares `automatic` with `reference` for backend benchmarks
and with `scalar` for kernel benchmarks.

## Troubleshooting and maintenance

A successful Gradle task without fresh JSON can mean a stale JMH lock. `report.sh` checks for JSON newer than its marker and reports a detected lock; confirm no benchmark is running before removing a stale lock.

When adding or changing a public numerical API, update `public-numerical-api.tsv` and `benchmark-coverage.tsv`, then
run `./gradlew :koblas-bench:checkBenchmarkCoverage`. The inventory names each reviewed public operation and maps it
to a benchmarked manifest row or an explicitly justified exclusion. The check fails for a public operation absent
from the inventory, an inventory entry without a manifest row, an unlisted benchmark method, or a manifest method
that no longer exists.
