# koblas-bench

CPU benchmarks for Koblas, OpenBLAS, Accelerate, and oneMKL. Requires JDK 25; the koblas C kernel arms also
need a C compiler. Vendor arms call each library through the production Koblas bindings on both runtimes, so a
vendor row reports the route of the call that was timed.

```bash
# Default capture: JVM scalar, C, SIMD, native, and available platform vendors.
koblas-bench/capture-report.sh --samples 5 --warmups 5 --target-ms 200 --forks 2

# Coarse dot sweep across all targets; short timings for exploration only.
koblas-bench/capture-report.sh --suite sweep --operation dot \
  --samples 1 --warmups 0 --target-ms 1 --forks 1 --output build/dot-sweep

# First three selected cases, one short sample, no warmup.
koblas-bench/capture-report.sh --smoke

# Vendor-only run.
koblas-bench/capture-report.sh --vendors-only --libraries openblas,accelerate
```

Default full reports go to `reports/<hardware-sha256>/`; single-kernel, smoke and vendor-only runs default to
`build/benchmarks/<hardware-sha256>/`. Identical hardware configurations share a directory. The key hashes
sorted hardware facts, excluding Linux's boot-dependent usable-memory count; hostname and date are not part of it.
Each successful run replaces the previous report in that directory. Use `--output DIR` to override.
Each report contains only one CSV per target and `metadata.txt` with hardware,
source revision, timing settings, execution timestamps, and actual runtime/library configurations.
Each CSV records one row per case with sample/fork counts, median, minimum, and maximum ns/op.
CSV files contain only case definitions, status, actual kernel and measured results; run settings and runtime
identity belong exclusively to `metadata.txt`. Capture is the only reporting script.

Use `--native-variant scalar|sse2|avx2|neon` for exact raw C arithmetic in the JVM C and Native targets.
Unavailable variants fail the capture. Kotlin scalar and JVM SIMD targets retain their distinct identities.

[`cases.txt`](cases.txt) defines every workload and its selection membership. Ordinary captures run the
163 default cases; `--operation NAME` intersects that suite with a single kernel. Use `--suite sweep`
with a specific `--operation NAME` to select its opt-in sizes. Sweeps currently cover `dot`, `sum`, `asum`,
`ssqd`, `dot4`, and `dot-axpy`, with 12–15 explicit sizes per operation. `--smoke` takes the first three
cases **after** both filters. An unknown suite, a sweep without an operation, or an empty intersection fails.

Untagged cases belong to `default`; `+suite=sweep` is opt-in, and `+suite=default,sweep` shares a single
case between both suites. A workload is listed once. Membership is selection metadata and is excluded
from canonical case IDs, fixture generation, timing boundaries, and CSV comparison fields. Changing only
membership leaves historical comparisons valid. `metadata.txt` records the requested suite, operation,
and selected case count alongside source revision and runtime identity.

For direct Gradle runs, use `-Pbench.suite=sweep -Pbench.operation=dot` with any benchmark task and the
existing `bench.samples`, `bench.warmups`, `bench.targetMs`, and `bench.forks` controls. Native and vendor
executables accept `--suite=sweep --operation=dot`. Omitting the suite always selects `default`.

Start tuning with the short coarse command above. If results suggest a crossover, add a few explicit
sizes near it in `cases.txt`, retaining each case's options and recipe. Confirm any decision with normal
warmups, samples, and forks, for example `--samples 5 --warmups 5 --target-ms 200 --forks 2`. The initial
sizes span tiny inputs, SIMD tails, current host-call neighborhoods, and larger working sets; they are
fixed workload entries, never generated from runtime widths or dispatch thresholds. Add future vector
or matrix sweeps the same way, varying useful options independently. Short exploratory timings are
not evidence for changing production thresholds.

Fixed packed recipes still require matching geometry; logical comparisons still require matching logical
inputs and timing boundaries, with physical recipe differences made explicit.
Compare matching cases and timing boundaries; prepared, one-shot, and packing-inclusive timings differ.
New references must match the fixture, validation, buffer reuse, and numerical contract.
Unsupported cases have no timing. A selected target failure stops capture and preserves the previous report.

For `scal` and `spgather`, `+timing=arithmetic` excludes resets; arithmetic scaling uses alpha = -1.
Default gather resets only the dense source; older captures also reset the sparse output, so compare those separately.

Each vendor contributes two targets: `<vendor>` through the Native binding, which reaches the library from
native code with nothing between the caller storage and the call, and `<vendor>-jvm` through the JVM binding,
which copies operands into native memory and includes that transfer in the timing.

Every BLAS invocation uses one compute thread. The binding establishes that at load, before any arithmetic, and
refuses a library it cannot hold to one thread; there is no thread setting to pass. `--libraries all` selects
OpenBLAS and Accelerate on macOS, OpenBLAS and oneMKL on Linux.
Each binding resolves its own library and reports the file the symbols actually came from, so a row naming a
vendor names the file that ran. The oneMKL candidates cover the loader path and the standard oneAPI prefixes,
because the installer puts the library somewhere the loader does not search. Only the dense CBLAS surface is
bound: Level 1 extensions, the fused panel kernels, the packed cases and everything sparse have no vendor entry
point and are reported unsupported with a reason rather than timed through a substitute.
