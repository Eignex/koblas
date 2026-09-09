# koblas-bench

Development benchmarks for every reviewed public numerical operation in koblas. This module is not published.

## Contribute a hardware report

```bash
koblas-bench/report.sh preflight jvm
koblas-bench/report.sh smoke jvm
koblas-bench/report.sh standard jvm
```

`preflight` reports the actual Gradle-selected benchmark runtime, host capabilities, and available comparator
libraries without running measurements. `smoke` builds once and executes two short fresh passes over two setup
cases; it validates the runner and implementation resolution but is not performance evidence. `standard` runs
the versioned 83-case built-in workload twice. It includes small and rectangular dense work, packed tile edges,
and prepared and one-shot sparse work. The runner prints the case count before it starts and the measured elapsed
time when it finishes.

As a calibration rather than a promise, the standard JVM workload plus the matched OpenBLAS arm completed in
245.1 seconds on a 12th Gen Intel Core i9-12900H using four pinned logical CPUs. Host load, toolchain caches, and
optional comparator availability can materially change that time.

The default needs no external numerical library. Add an explicitly matched comparator or probe all optional
comparators with:

```bash
koblas-bench/report.sh standard jvm --comparators openblas
koblas-bench/report.sh standard jvm --comparators onemkl
koblas-bench/report.sh standard jvm --comparators auto
```

An explicitly requested missing or unsupported comparator fails before the build. `auto` records every capability
and runs only available arms; it never substitutes one implementation for another. Built-in-only reports provide
hardware coverage, not an external parity claim. Comparator processes use the identical case IDs, seed, warmups,
measurements, and single-thread settings as their built-in matches. Linux affinity can be recorded and applied with
`--cores 2-5`; it does not reserve the machine. Existing tuning settings are recorded, and a deliberate override
can be supplied as `--tuning KOBLAS_DENSE_GEMM_SMALL=64`.

| target | built-in | OpenBLAS | oneMKL |
| --- | --- | --- | --- |
| JVM on Linux/macOS | yes | detected at runtime | detected at runtime |
| Linux x86-64 Native | yes, no external library required | detected at runtime | unsupported |
| macOS Apple Silicon Native | yes | unsupported | unsupported |

OpenBLAS needs a loader-visible `libopenblas.so.0`, `libopenblas.so`, or `libopenblas.dylib`. oneMKL needs a
loader-visible `libmkl_rt`. The Linux Native binding opens OpenBLAS dynamically, so the built-in executable neither
links nor requires it. Native oneMKL and macOS Native external bindings are not implemented.

Each invocation owns a UUID-named output tree and never deletes or scans another run's files. Gradle performs one
build, then executes fresh passes into distinct directories. JMH's advisory process-lock rejection is disabled for
these runs; concurrent runner activity is recorded as noisy evidence rather than blocked or deleted. Fork errors,
resolved identities, allocation checks, exact expected case counts, pass IDs, raw JSON, and complete logs remain in
the bundle.

The resulting `.tar.gz` contains versioned metadata and rows, stable case IDs, raw passes and logs, both coverage
manifests, readable `summary.txt`, CSV/JSON rows, and SHA-256 checksums. Uncertainty combines reported benchmark
error with cross-pass variation. A `koblas/comparator` timing ratio is emitted only when case ID, workload version,
settings, and units match. Unknown metadata stays `unknown`; structured metadata omits hostname, username, and the
checkout path. Logs can still contain local paths, so inspect them before sharing.

Validate or summarize a bundle offline:

```bash
koblas-bench/report.sh validate path/to/koblas-hardware-jvm-....tar.gz
koblas-bench/report.sh summarize path/to/koblas-hardware-jvm-....tar.gz
```

To submit results, open a benchmark-results issue, paste `summary.txt`, describe any known competing load, and
attach the inspected archive manually. There is no upload command. A maintainer can validate it offline and, when
accepted as project evidence, archive it with a PR under `koblas-bench/results/` and update
[`results/README.md`](results/README.md).

## Run benchmarks

```bash
./gradlew :koblas-bench:jvmFullBenchmark
```

The `full` configuration expands every declared parameter, including optional external arms, and therefore requires
all corresponding runtime libraries. It is exhaustive developer coverage, not the bounded contributor workload.

The benchmark runner can report a fork failure after Gradle itself has completed successfully. Treat any
`<failure>` or `EXCEPTION: <ERROR>` line as a failed comparison; the contributor runner enforces this automatically.

For local A/B work:

```bash
./gradlew :koblas-bench:jvmSelectedBenchmark \\
  -Pbench.include='Level3Benchmark.gemm|SyrkBenchmark.syrk|Syr2kBenchmark.syr2k' \\
  -Pbench.param.n=256 \\
  -Pbench.param.denseArm=built-in,openblas
```

### Arms

Dense parity uses `denseArm=built-in,openblas,onemkl`; retained sparse BLAS uses
`sparseArm=built-in,onemkl`. These arms construct the built-in implementation or open a benchmark-owned external
binding directly. They never use production discovery, and every setup asserts and reports its resolved identity.

Kernel microbenchmarks use `kernels=built-in,scalar,c,simd`. `built-in` selects the immutable platform engine;
the other names are exact pins and fail when unavailable. Dense and sparse comparator arms use the same
`built-in`, `openblas`, and `onemkl` terminology.

Benchmark-owned level-1 CBLAS calls cover dot, axpy, scale, norm, absolute sum, swap, and rotations for both dense
comparators. OpenBLAS `cblas_dsum` is tracked as a vendor extension; fused squared distance has no CBLAS
counterpart. Four-way dot is labeled as a composition of four calls, never direct kernel parity. See
`comparator-coverage.tsv` for the same direct/composition/missing classification across dense and sparse operations.

Comparator equivalence is separate from standards classification. See [`operation-classification.md`](operation-classification.md)
for standard, matrix-property, legacy indexed, vendor extension, composition, and private-fusion categories.

`simd` is absent from every `@Param` list because Kotlin/Native has no such
arm and a benchmark configuration covers every target, so a full native
run would ask for one that cannot exist. Pass it explicitly on the JVM.

Every arm checks what it actually resolved to and fails when that is not what
the arm names, so a run cannot quietly credit an implementation that never
executed. Setup prints a `resolved: arm=...` line naming the immutable engine,
and the contributor runner collects those lines into the report.

`SparseWorkspaceBenchmark` measures the caller-owned sparse support operations directly at small through large
touched counts. `SparseWorkspaceGrowingScatterBenchmark` performs many short scatters while support grows, exposing
any per-call dependence on retained support. The comparison suites add contract-equivalent competitors rather than
claiming that one vendor call implements the workspace API. `scatterAxpy` composes oneMKL `cblas_daxpyi` with
validation, epoch marking, stale-entry initialization, and ordered first-touch tracking. `gatherTouched` composes
`cblas_dgthr` with ordered index emission and optional exact-zero compaction; `gatherClearTouched` similarly composes
`cblas_dgthrz` with mark clearing. Destructive fixture restoration is included in those rows and reported separately,
so repeated measurements never gather an already-cleared zero state. Primitive-only rows are explicitly partial
work. Independent Kotlin rows cover active maximum, pivot candidates, and checked scatter diagnostics where there is
no vendor counterpart.

The raw slice ABI and arithmetic follow Intel's official
[`cblas_?axpyi`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/cblas-axpyi.html),
[`cblas_?gthr`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/cblas-gthr.html),
and [`cblas_?gthrz`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/cblas-gthrz.html)
contracts. The composed scatter timing deliberately uses finite values and finite nonzero alpha: exceptional and
zero-alpha IEEE diagnostics belong to `scatterAxpyChecked` and are covered by the independent baseline, because
oneMKL does not expose the required per-product flags.

These developer comparisons are intentionally outside the immutable contributor profile v1. Run a bounded local
comparison with `jvmSelectedBenchmark`, setting `sparseArm=built-in,onemkl` for the composed rows and
`baselineArm=built-in,baseline` for the independent rows. A requested missing oneMKL arm fails; Native oneMKL remains
unsupported. `ExplicitPackedKernelBenchmark`
selects scalar and bundled C arms so full and logical-edge tiles, fused update/solve, and the explicit
composition remain distinguishable. On the JVM, `PackedTrsmEligibilityBenchmark` isolates the conservative eligibility scan and
bound, while `ExplicitPackedTrsmBenchmark` keeps its outcome inside a repeated end-to-end solve.

Sparse one-shot rows include oneMKL CSC conversion and destruction. Prepared rows retain the inspector-executor
handle across invocations. Fresh sparse results and packing are reported with workload-dependent allocation
expectations; allocation-free JVM kernels are probed in every fork and invalidate it if managed allocation exceeds
the near-zero allowance.

## Troubleshooting and maintenance

Contributor runs use distinct report and Native-description directories, accept concurrent JMH processes, and
never remove `/tmp/jmh.lock` or another invocation's files. If an ordinary developer task reports a lock, inspect
the owning process; do not delete a live run's lock.

When adding or changing a public numerical API, update `public-numerical-api.tsv` and `benchmark-coverage.tsv`, then
run `./gradlew :koblas-bench:checkBenchmarkCoverage`. The inventory names each reviewed public operation and maps it
to a benchmarked manifest row or an explicitly justified exclusion. The check fails for a public operation absent
from the inventory, an inventory entry without a manifest row, an unlisted benchmark method, or a manifest method
that no longer exists.
