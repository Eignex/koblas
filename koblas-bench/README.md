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
| `jvm-default` | The engine ordinary JVM calls use, with SIMD enabled |
| `jvm-default-no-simd` | The default engine without the Vector API module; vendor access stays enabled |
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

Prepared Native executables belong to each revision's verified build workspace. Shared lab inputs
contain workload data and external vendor packages; a revision's executable is recorded as the
`nativeExecutable` case attribute and its path and SHA-256 are retained in `native-program.txt`.
This permits paired revisions whose programs differ while keeping workload input bytes identical.

For JVM compiler diagnostics, `lab/cases-jit-policy.txt` samples vector casts, product edges,
retained scratch and sparse scheduling. Compare assertions on/off separately under normal tiering,
`-XX:+TieredCompilation -XX:TieredStopAtLevel=1` and `-XX:-TieredCompilation`.
For logs attributed to the measured JMH fork, the lab parameter `profile-compilation=true`
(`KOBLAS_BENCH_PROFILE_COMPILATION=true` locally) retains `*-jit-*.xml` files beside each target's
CSV. These HotSpot logs record the compiler, compilation level and inlining decisions separately
for every fork. Instrumented timings are not performance evidence, and this instrumentation is
refused in the fixed status-regression protocol. Compare policy changes within each compiler regime;
non-tiered C2 collects profiles differently from normal tiering.

## Details

- [Capture reference](REFERENCE.md): options, workload definitions, attribution, host setup, lab adapter and scoring.
- [Case inventory](cases.txt): available workloads and sizes.
- [Engine guide](../BACKENDS.md): library setup, automatic selection and native thresholds.

## Per-commit status sweep

`lab/recipes/status-sweep-jvm.json` runs `jvm-default`, `jvm-simd`, `accelerate-jvm` and
`jvm-scalar` together for each workload on the frozen **status-v3** inventory:
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
profile change establishes a fresh baseline. Timing controls remain fixed. Multi-engine rows use a separate `-engines-v1` regression suite and
export `median_ns` for `jvm-default` plus `median_ns/<target>` for the other engines. Every metric retains
its requested engine, measured provider and actual kernel; SIMD-arm measurements may reach scalar
fallbacks. The exact Accelerate arm records unsupported operations as untimed gaps, including sparse
operations and sum; those gaps never acquire portable timings. The optional five-engine recipe,
`lab/recipes/status-sweep.json`, also covers portable and default Kotlin/Native. Unsupported exact-engine view adapters retain their unsupported
status and contribute no invented timing or regression coverage.

The recipe matches the local Mac arm64 platform used by Klause's status sweep. It checks `main` every 120
seconds, runs at priority 98 with four vCPUs (up to three single-thread cases), and requests an 1800-second
active-time ceiling including preparation for the four JVM arms. Queue waits and pauses are outside that ceiling. The recipe requests three
independent repetitions using `repeats=3` and `repeatOrder=pass`: run the full inventory once before
starting the next repetition.
Each workload's engine arms stay on one machine in one worker attempt. `arm-order=rotate` rotates the
first engine across workloads; it does not split arms across workers. Every JVM arm has one JMH fork,
one round, three 100ms warmup iterations and three 100ms measurement iterations, Serial GC and a fixed
512MiB heap. The JVM recipe prepares only JVM classes and libraries. The optional five-engine recipe
also builds and freezes the Mac arm64 Native executable. The trend uses the median
of the three independent capture medians. Raw captures and failures remain
available; incomplete repetitions or changed controls cannot establish an aggregate baseline. The JMH
iterations within one capture remain consecutive; the three independent captures are spread across the run.
This is a short change detector; use longer paired captures to measure small differences.

Each case validates numerical results against a scalar reference before timing, including complete backing
buffers for indexed sparse mutations, retains the raw CSV/logs and exports `median_ns` in ns/op.
A numerical/capture failure, a missing engine report or an unsupported default-engine workload fails its case. The lab's `metric-v1`
confirmation policy selects correctness transitions and timing changes beyond 15%, reuses both accepted revision
builds and frozen inputs, and runs three paired blocks at priority 99. Each block runs both revisions on one machine
in one attempt; other rows may use other compatible local machines. Confirmation repetitions also run as
full inventory passes, and revision arm order rotates between passes. A majority must agree. Changed controls,
missing samples, instrumentation and skipped cases cannot vote. At most 40 affected workloads are confirmed per
run, prioritizing correctness transitions; the rest stay unconfirmed. Retain failed-attempt costs separately.

The report plots JVM SIMD candidate regression counts and performance against each workload's own
fixed baseline, including theme panels. The case table keeps absolute ns/op and each engine's own
historical comparison separate. Paired confirmations compare their recorded neighboring revisions and vote independently
for each exported engine metric; their status is kept distinct from fixed-baseline candidates.

The regression charts deliberately show exact `jvm-simd` measurements. `jvm-default` is an automatic
selection policy and cannot populate that line, even when some of its operations select SIMD. The case
table shows independently measured JVM SIMD, Accelerate and portable results alongside JVM Default.
Missing or unsupported exact-engine arms remain explicit gaps.

Before changing the installed schedule, deploy coordinator support for `repeatOrder=pass`, repeated
workload median aggregation, `regressionMetrics` and engine-aware trend fields. Pilot the actual JVM
comparison recipe on Mac arm64 and record preparation, active time and failures before installing it.
Keep the three inventory passes and paired workload rows; the previous 600-second cap was budgeted
for one default-engine arm.
`lab/recipes/status-sweep-default.json` retains that older protocol. The optional five-engine Native recipe
still requires a separate pilot and is not enabled by the JVM comparison schedule.

The four-arm JVM recipe was piloted on 12 November 2026 on Mac arm64 with JDK 25 in
[lab run 11869](https://lab.eignex.com/jobs/11869): all 408 paired captures succeeded in 981.031 seconds
of active time under the 1800-second ceiling. Preparation took 14.590 seconds with a build-cache miss;
row durations ranged from 2.530 to 9.260 seconds, with a 7.128-second median. No preparation or
measurement attempt failed. The captures provide three measurements each for 136 default-engine,
133 exact SIMD, 133 portable and 77 Accelerate workloads. Unsupported exact-engine view adapters
and vendor gaps retain their status without a timing. The accepted plan separates repeated captures
of each workload by 136 rows and rotates engine order across workloads.

The default-only three-pass recipe was piloted on 11 November 2026 on an Apple M4 with JDK 25 in
[lab run 11830](https://lab.eignex.com/jobs/11830): all 408 captures succeeded in 371.139 seconds of
active time under the 600-second ceiling. Successful preparation took 15.247 seconds with a verified
build-cache hit; measured row durations ranged from 1.835 to 4.340 seconds, with a 2.226-second median.
Two earlier preparation attempts failed because pinned workers rejected the coordinator's new ordering
field. Those failures remain recorded separately; worker registration now omits that scheduling-only
field. The accepted plan places 136 rows between repeated captures of each workload. This budgets the
default-only recipe; the five-engine recipe still needs its own pilot.

After deployment and the pilot, register it with the maintained HTTP client:

```bash
lab schedule koblas-sweep main koblas-bench/lab/recipes/status-sweep-jvm.json 120
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
are part of the cost beyond the nominal 600ms of iterations per row. Use the measured budget for the selected engine recipe; a default-only pilot does not budget the
additional comparison arms. A local capture of the whole inventory validates cases, but does
not reproduce the lab's per-row startup and placement costs. The documented confirmation cap remains 40 rows;
a broader change can leave additional affected rows unconfirmed and needs follow-up from the lab owner.
