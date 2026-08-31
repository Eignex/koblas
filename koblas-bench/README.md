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

`automatic` measures normal production discovery, `reference` installs the
portable implementation, and `host` explicitly installs a host backend and
fails if unavailable. The report profile compares `automatic` with `reference`
for backend benchmarks and with `scalar` for kernel benchmarks.

## Troubleshooting and maintenance

A successful Gradle task without fresh JSON can mean a stale JMH lock. `report.sh` checks for JSON newer than its marker and reports a detected lock; confirm no benchmark is running before removing a stale lock.

When adding or changing a public numerical API, update `benchmark-coverage.tsv` and run `./gradlew :koblas-bench:checkBenchmarkCoverage`. It fails for either an unlisted benchmark method or a manifest method that no longer exists.
