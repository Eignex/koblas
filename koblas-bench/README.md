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

For copy tuning, `--cases koblas-bench/lab/cases-copy-throughput.txt` covers short and streaming
lengths, strided fallbacks, overlapping views and an unchanged dot-product control.

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
selected engine; pinning an exact engine through `KoblasEngine` is what makes an arm exact, and an
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

`lab/recipes/status-sweep.json` runs the default JVM engine on the frozen **status-v3** inventory:
136 representative cases, retaining all 82 v2 identities. It covers the engine routine families and samples
vector crossover neighbors, odd tails, rectangular and transposed products, upper triangles, right-side
operations, unit diagonals, strided vectors, borrowed GEMM storage and retained scratch. A borrowed product
crosses all three product cache block extents, and triangular cases cross the diagonal block with few right-hand
sides. Sparse cases sample three densities and uniform, banded, skewed, empty-column and mixed support.

This is a bounded change detector. It deliberately samples independent risk factors rather than taking their
Cartesian product or counting every public overload. [The v3 inventory](lab/cases-status-sweep-v3.txt) records
membership and intent; tests protect representative coverage and frozen identities. The lab parameter
`sweep-profile=status-v2` retains the old inventory, `status-v3` selects the core, and `extended-v1` selects the
broader inventory. A profile and a custom `cases` path are mutually exclusive. Status requests default to v3;
profile names are carried into case parameters, regression suite attributes and measurement controls so a
profile change establishes a fresh baseline. Timing controls remain fixed.

The recipe matches the local Mac arm64 platform used by Klause's status sweep. It checks `main` every 120
seconds, runs at priority 98 with four vCPUs (up to three single-thread cases), and has a 600-second active-time
ceiling including preparation. Queue waits and pauses are outside that ceiling. Each workload has one JMH fork,
one round, three 100ms warmup iterations and three 100ms measurement iterations, Serial GC and a fixed 512MiB
heap. This is a short change detector; use longer paired captures to measure small differences.

Each case validates numerical results against a scalar reference before timing, including complete backing
buffers for indexed sparse mutations, retains the raw CSV/logs and exports `median_ns` in ns/op.
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

## Extended sweep

[Extended v1](lab/cases-sweep-extended-v1.txt) contains the 136 core cases plus 51 selected cases for larger
working sets, additional vector crossovers, strided storage and matrix orientations. It is an opt-in capture;
the recipes do not install schedules:

- [Mac arm64](lab/recipes/sweep-extended-mac-arm64.json) compares `jvm-default`, `jvm-scalar` and `jvm-simd`.
- [Linux x64](lab/recipes/sweep-extended-linux-x86_64.json) also compares `native` and `native-default`.

Each workload keeps every selected arm together on one compatible machine and worker attempt. Placement is
automatic and may change between rows. Unsupported view adapters appear only on the default JVM arm and
cannot contribute a complete paired comparison across exact-engine arms. These captures retain raw reports;
they use the ordinary comparison protocol rather than the short status regression policy. They use 20 warmups,
five samples, 200ms iterations, two JVM forks and three rotating rounds.

For a local capture, use the inventory directly:

```bash
koblas-bench/capture-report.sh --targets jvm-default,jvm-scalar,jvm-simd \
  --cases koblas-bench/lab/cases-sweep-extended-v1.txt \
  --rounds 3 --samples 5 --warmups 20 --target-ms 200 --forks 2 \
  --output koblas-bench/build/benchmarks/extended
```

Before deploying a changed core recipe, pilot the actual prepared row protocol on its target platform. Record
preparation time, total active time, row duration spread and failures; capture/JVM startup and reference checks
are part of the cost beyond the nominal 600ms of iterations per row. Keep the 600-second core ceiling until a
pilot demonstrates that it needs changing. A local capture of the whole inventory validates cases, but does
not reproduce the lab's per-row startup and placement costs. The documented confirmation cap remains 40 rows;
a broader change can leave additional affected rows unconfirmed and needs follow-up from the lab owner.
