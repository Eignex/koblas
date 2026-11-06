# koblas-bench

CPU benchmarks comparing Koblas engines and native BLAS libraries. Requires JDK 25.
Run commands from the repository root.

## Run a capture

```bash
# Quick integration check; smoke timings are not performance evidence.
koblas-bench/capture-report.sh --targets jvm-scalar,jvm-simd --smoke \
  --output koblas-bench/build/benchmarks/smoke

# Compare scalar and SIMD dot products over three rounds.
koblas-bench/capture-report.sh --targets jvm-scalar,jvm-simd --operation dot \
  --rounds 3 --samples 5 --warmups 5 --target-ms 200 --forks 2 \
  --output koblas-bench/build/benchmarks/dot
```

Use `--suite sweep --operation dot` for the larger dot size sweep, or `--cases PATH` for a custom
[case file](cases.txt). `capture-report.sh --help` lists the available options.
On Linux, `--cpu spread` chooses and pins a lightly loaded allowed core; it does not reserve it.

| Target | Measures |
|---|---|
| `jvm-scalar` | Portable Kotlin |
| `jvm-simd` | JVM SIMD with its scalar fallbacks |
| `jvm-default` | The engine ordinary JVM calls use |
| `native`, `native-default` | Portable and platform-default Kotlin/Native engines |
| `<vendor>`, `<vendor>-jvm` | Native and JVM vendor bindings; JVM timings include array transfers |

Vendor names are `openblas`, `onemkl`, `aocl`, `armpl` and `accelerate`; choose those available on the host.
For example, `--libraries onemkl --targets jvm-simd,onemkl-jvm` compares JVM SIMD with the JVM oneMKL
binding. Target availability and Native cross-compilation are covered in the [reference](REFERENCE.md).

## Read the results

Captures write one CSV per target plus `metadata.txt` into the chosen output directory. Repeated
captures keep each round under `pass-1`, `pass-2`, and so on. Engine rows record timing and the
selected engine; pinning an exact engine through `BuiltinEngines` is what makes an arm exact, and an
engine's name alone does not establish that a given size vectorized.

Compare matching cases and settings on the same host. For repeated captures, take the median of each
case's round medians, inspect the spread, and check unchanged controls. Setup validates results against
scalar references before timing. Workloads differ in whether reset or allocation is timed;
check their [definitions](REFERENCE.md#cases) before comparing them.

## Details

- [Capture reference](REFERENCE.md): options, workload definitions, attribution, host setup, lab adapter and scoring.
- [Case inventory](cases.txt): available workloads and sizes.
- [Engine guide](../BACKENDS.md): library setup, automatic selection and native thresholds.

## Per-commit status sweep

`lab/recipes/status-sweep.json` runs the default JVM engine on 41 fixed arithmetic workloads spanning the
`KoblasEngine` routine families at exactly two sizes (82 cases). Typed storage overloads share these
underlying kernels; the sweep checks routine-family coverage without counting public overloads. `lab/cases-status-sweep-v2.txt` freezes membership. Dense vectors and indexed
vectors use 64/4096 entries, dense Level 2 and sparse matrices use orders 32/128, and dense Level 3 uses
16/64. Sparse density is 1/8; matrix panels have eight right-hand sides.

The recipe matches the local Mac arm64 platform used by Klause's status sweep. It checks `main` every 120
seconds, runs at priority 98 with four vCPUs (up to three single-thread cases), and has a 600-second active-time
ceiling including preparation. Queue waits and pauses are outside that ceiling. Each workload has one JMH fork,
one round, three 100ms warmup iterations and three 100ms measurement iterations, Serial GC and a fixed 512MiB
heap. This is a short change detector; use longer paired captures to measure small differences.

Each case validates numerical results before timing, retains the raw CSV/logs and exports `median_ns` in ns/op.
A numerical/capture failure or an unsupported default-engine workload fails its case. The lab's `metric-v1`
confirmation policy selects correctness transitions and timing changes beyond 15%, reuses both accepted revision
builds and frozen inputs, and runs three paired blocks at priority 99. Each block runs both revisions on one machine
in one attempt; other rows may use other compatible local machines. A majority must agree. Changed controls,
missing samples, instrumentation and skipped cases cannot vote. At most 40 affected workloads are confirmed per
run, prioritizing correctness transitions; the rest stay unconfirmed. Retain failed-attempt costs separately.

After deploying lab support for `metric-v1`, register it with the maintained HTTP client:

```bash
lab schedule koblas-sweep main koblas-bench/lab/recipes/status-sweep.json 120
```
