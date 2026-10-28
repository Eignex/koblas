# koblas-bench

CPU benchmarks for Koblas and the vendor libraries it selects: OpenBLAS, oneMKL, AOCL, ArmPL and Accelerate.
Requires JDK 25. `capture-report.sh` supplies each measured capture.

The JVM harness and annotation processor are built together from pinned
[OpenJDK JMH sources](https://github.com/openjdk/jmh/tree/a194eead0136bb66e5e59e4fdb2e18543e730929).
This revision removes the deprecated `sun.misc.Unsafe` field-layout probes still present in JMH 1.37.
The source archive is checked by SHA-256, and reports carry the JMH version and revision. This harness is
an unreleased upstream build; historical 1.37 reports retain their original attribution. The JMH subproject
supplies only benchmark tooling and publishes no library artifacts.

## Running a capture

```bash
# Full capture: three JVM arms, two Native arms, and the platform's vendors.
koblas-bench/capture-report.sh --samples 5 --warmups 5 --target-ms 200 --forks 2

# One operation across its opt-in sizes.
koblas-bench/capture-report.sh --suite sweep --operation dot --samples 5 --warmups 5 --target-ms 200 --forks 2

# First three selected cases, one sample, no warmup.
koblas-bench/capture-report.sh --smoke

# Vendors only.
koblas-bench/capture-report.sh --vendors-only --libraries openblas,accelerate
```

For a retained three-round reduction capture on a Linux lab node:

```bash
koblas-bench/capture-report.sh --libraries openblas,onemkl --targets jvm-simd,openblas,onemkl \
  --cases koblas-bench/cases-reductions.txt --rounds 3 --cpu auto \
  --samples 5 --warmups 5 --target-ms 200 --forks 2 --output /path/to/new-report
```

Each round rotates the target order and retains its own CSVs, metadata and log under the new output
folder. Source hashes must stay unchanged, and binaries must match after every completed round.
For each case and target, take the median of the round `median_ns` values before calculating throughput;
the round files retain the spread. `--cpu auto` pins the capture and its child processes to the first
CPU allowed by the job's affinity mask. It does not isolate that CPU from other work.

Lab jobs use the [Bench SDK adapter](#bench-sdk-adapter) to prepare and run these captures.

On a hybrid CPU, pin the run: `taskset -c 2,4,6,8 koblas-bench/capture-report.sh …`. Otherwise it wanders
between core types and the timings move with the scheduler.

Lab workers may run a source archive without `.git`. Such a capture records `source_kind=archive`,
with `source_commit` and `dirty` unavailable; use the lab job's pinned revision for attribution.
Multi-round archive captures fingerprint every source file except generated directories (`build`,
`.gradle`, `.kotlin`, `.idea`, `.git` and `node_modules`) before and after each round, alongside the
same binary hashes as a Git capture. This allows the lab archive to run without inventing Git metadata.

### Comparing JDK runtimes

The benchmark launcher defaults to JDK 25. `-Pbench.javaVersion=27` selects JDK 27 for
measurement while compilation keeps its configured toolchain. `-Pbench.expectedJavaHome=/path/to/jdk`
pins the launcher to that installation. Captures record the measured runtime in their target logs.

For collector comparisons, `capture-report.sh` accepts `--jvm-gc g1|serial|zgc` and
`--jvm-heap 1g`. Without these options, runtime defaults remain part of the measurement;
the results do not isolate JIT code generation.

## Shrinking rank updates

Three default operations compare the trailing updates of a small dense LU workload:

| Operation | Timed work |
|---|---|
| `ger-shrinking` | `n - 1` full-buffer `ger(-1.0, x, y, A)` calls with zero prefixes |
| `ger-window-shrinking` | The same updates through the validated windowed GER API |
| `panel-rankupdate-shrinking` | The same rectangles passed directly to raw panel kernels |

Orders 5, 6 and 7 bracket the consumer's dense-selection boundary; 34 and 198 are larger controls, outside
its production dense path. Order 6 is primary: a synthetic matrix with 31 nonzeros out of 36, then five
updates with vector supports 5, 4, 3, 2 and 1. Mixed-sign coefficients avoid an all-positive special case.
These fixtures reproduce the workload shape and density, rather than a captured basis's coefficients.

Matrix and vector buffers are disjoint and reused. Each support has a prepared vector pair, so timing
includes warmed call dispatch and arithmetic, and excludes vector preparation, matrix resets, allocation
of fixtures, pivot search, swaps, division and sparse conversion. A sample repeatedly accumulates the
batch into the same matrix. Setup checks the whole matrix against the scalar oracle and restores it before
timing. A row reports nanoseconds **per batch**, not per individual GER or complete factorization.
The raw panel row omits public validation and alias handling; it isolates panel work from those public API costs.
Routes name every body reached by the shrinking windows, including scalar tails.

```bash
# Repeat for ger-shrinking and panel-rankupdate-shrinking to compare the three paths.
taskset -c 2 koblas-bench/capture-report.sh --operation ger-window-shrinking \
  --samples 5 --warmups 5 --target-ms 200 --forks 2
```

Explicit vendor arms support only the full-buffer batch, since the binding API has no rectangle offsets
or parent leading dimension. A host-composed engine keeps windowed GER on its portable schedule. Capture
GC allocation data separately with JMH profiling when investigating JVM dispatch; the ordinary CSV has
no allocation column. The JVM verification gate checks the warmed five-window batch for allocations.
These GER-only timings cannot establish a complete LU speedup.

## Simplex unit AXPY

`axpy-unit` measures the contiguous raw `vectorKernels.axpy` call with `alpha = 1`, matching
Klause's cached basic-value restoration. `lab/cases-simplex-axpy.txt` includes the retained basis-corpus
row counts 2, 6, 8, 27, 56 and 70, a 64-element control, and existing nonunit AXPY controls.
These dimensions are workload examples, not a production call-frequency distribution.

Disjoint arrays are reused and the destination is not reset during timing. Repeated accumulation stays
finite over the capture; it excludes the solver's FTRAN and cache-restoration work. Setup checks two
successive complete destinations against the explicit scalar engine and restores the initial destination.
Vendor arms perform the same accumulation; the JVM vendor arm includes operand transfers on every call.
The existing `axpy` operation retains its nonunit multiplier and timed destination reset, so its absolute
times must not be compared as if it were the same workload.

## Selected CSC column dots

`spdot-column` and `spdot-column-raw` price the same column at a nonzero offset through
`SparseKernels.dotColumn` and the validated raw `dot` overload. The four default cases store 8, 64, 512
and 4096 entries. Both include public validation and the indexed leaf; the column API reuses validated
support while the raw API checks every index. Construction and dense-vector setup are outside timing.
Each case is checked against the scalar indexed dot before measurement. Neighboring columns contain NaNs
to catch a window that reads outside its column.

These composed rows name the actual indexed leaf even when a SIMD engine selects a scalar fallback.
Compare matching cases on the same arm; an engine label alone does not establish vectorization.

```bash
# Repeat with spdot-column-raw for the matched raw path.
taskset -c 2 koblas-bench/capture-report.sh --operation spdot-column \
  --samples 5 --warmups 5 --target-ms 200 --forks 2
```

These are repeated-call API measurements. They exclude matrix construction and solver work, so a consumer
needs a separate setup-inclusive solver A/B before claiming an end-to-end improvement.

## Capture hosts

### Native on ARM64

The Kotlin/Native compiler ships no linux-aarch64 host, so an ARM64 machine cannot link the executable it
would time. It can still run one. Build it on an x86-64 host, where the compiler cross-compiles to the
target, carry it over, and name it:

```bash
# On an x86-64 machine, in a clone of the repository.
./gradlew :koblas-bench:linkReleaseExecutableLinuxArm64
scp koblas-bench/build/bin/linuxArm64/releaseExecutable/koblas-bench.kexe arm-host:

# On the ARM64 host.
koblas-bench/capture-report.sh --native-executable ~/koblas-bench.kexe \
  --samples 5 --warmups 5 --target-ms 200 --forks 2
```

Without it the capture is the JVM arms and `armpl-jvm` alone, which `native_capable=false` in `metadata.txt`
records. `native_executable` there says which binary produced the native rows, since one built on another
machine is not the same evidence as one linked where it ran.

Nothing about this is automatic, so an ARM capture that does not pass the flag silently covers less. The
`linuxArm64` Level 1 kernels are shipped production code, and the plain `armpl` arm is the binding a Kotlin/
Native consumer on that hardware actually uses.

### Which hosts to capture

The historical captures provide a ladder of SIMD widths and generations. The restored matrix kernels need
separate captures on these microarchitectures before any report can support performance claims about them.

| Microarchitecture | Widest SIMD | Double lanes |
|---|---|---|
| Sandy Bridge | AVX, no FMA3 | 2 |
| Ivy Bridge | AVX, no FMA3 | 2 |
| Haswell | AVX2 with FMA3 | 4 |
| Zen 2 | AVX2 with FMA3 | 4 |
| Sapphire Rapids | AVX-512 | 8 |
| Neoverse-N1 | NEON | 2 |
| Neoverse-V1 | SVE 256-bit | 4 |
| Neoverse-V2 | SVE 128-bit | 2 |

Lane counts are what `SPECIES_PREFERRED` resolved to on JDK 25, read back from each capture rather than
derived from the instruction set: the AVX hosts get 128-bit vectors because HotSpot caps its vector size
where only the first generation is present, and Neoverse-V2's SVE is narrower than Neoverse-V1's although it
is the later part. Every rung therefore has to be measured rather than predicted, which is the reason to keep
the list.

The pre-FMA rungs earn their place even though nobody buys those parts now: a Vector API operation with
no instruction behind it falls back to a software implementation per lane rather than refusing, and only a
host without the instruction shows that.

### Not yet covered

Further coverage gaps remain beyond the historical ladder.

The historical ARM rungs have no `native` or plain `armpl` numbers, for the build reason given above.
A cross-compiled executable can be carried to an ARM host for measurement. Compile-only validation does
not provide Native timing evidence.

The AMD rung is Zen 2, which stops at AVX2, so every AVX-512 number on the ladder is Intel's. AMD implements
that width differently enough that it is not the same rung read twice, and Zen 4 is where AMD has it. That
host matters more than a missing width would on its own, because AMD is also where `Vendor.select`
puts AOCL ahead of everything else, so it is the only host whose first-choice vendor is the one the `aocl`
arm times.

Accelerate and macOS are absent from this ladder, and Linux measurements cannot stand in for them.
Correctness checks on macOS are separate from performance measurements on Apple hardware. The Native default
uses portable Kotlin when an installed host call is unavailable or ineligible.

## Options

| Option | Effect |
|---|---|
| `--libraries openblas,accelerate,onemkl,aocl,armpl\|all` | Which vendors to run. `all` selects OpenBLAS plus oneMKL on x86-64 Linux, ArmPL on ARM64 Linux, or Accelerate on macOS. |
| `--suite default\|sweep` | Case suite. `sweep` requires `--operation`. |
| `--operation NAME\|all` | Intersects the suite with one kernel. |
| `--samples N`, `--warmups N` | Measured and discarded repetitions per case. |
| `--target-ms N` | Time budget per repetition. |
| `--forks N` | JVM forks. Native and vendor executables ignore it. |
| `--smoke` | First three selected cases at one short sample. |
| `--vendors-only` | Skips the Koblas targets. |
| `--output DIR` | Overrides the report directory. |
| `--native-executable PATH` | Runs the native arms from this binary rather than linking one. The only way to reach them on ARM64 Linux. |
| `--pass N` | Labels the run in `metadata.txt`. |

## Where results go

Full captures write to `reports/<hardware-sha256>/`; single-operation, smoke and vendor-only runs write to
`build/benchmarks/<hardware-sha256>/`. The hash covers sorted hardware facts, so identical machines share a
directory, and each successful run replaces what was there. A failing target stops the capture and leaves the
previous report in place.

A committed report comes from a machine kept for the purpose: nothing else running on it, a named part
someone else can rent or buy, and cores it does not share. Accelerate is reachable only on a dedicated Apple
machine that meets the same terms.

A machine being worked on does not qualify, whatever it is. Run captures there freely, since `--output` and
the single-operation modes write to `build/benchmarks/` anyway, but a browser on the other desktop is enough
to move the numbers: two runs here during recent work disagreed by a factor of two on lengths so large that
the loop waits on memory and the arithmetic cannot matter. A report like that describes the afternoon rather
than the library.

A report holds one CSV per target plus `metadata.txt`. The CSV has one row per case with sample and fork
counts and median, minimum and maximum ns/op, along with the kernel that actually ran. Run settings, hardware,
source revision and the resolved library files are in `metadata.txt` only.

## Targets

Koblas contributes independent `jvm-scalar`, `jvm-simd`, `jvm-default`, `native` and `native-default` arms.
`jvm-scalar` is portable Kotlin at every level, and `jvm-simd` is every Vector API kernel this library owns.
`jvm-default` is what an ordinary call gets, which is a policy rather than an exact arm. Its owned kernels
are the SIMD arm with the incubator module, and scalar otherwise. With native access enabled on Intel Linux,
an installed oneMKL can serve eligible Level 3 calls with every extent at least 128. Whole-matrix GEMM
also qualifies with both output extents at least 16 and depth from 1024 through 4096; borrowed views
retain the larger cutoff. The default and exact
SIMD arms therefore need separate measurements; each row names its route. The comparison that says what
Vector API kernels are worth is `jvm-simd` against `jvm-scalar`.

`native` and `native-default` are the same split on Kotlin/Native and are two different engines rather than
one named twice. `native` is the exact portable arm: common Kotlin at every level, resolving no library at
all, which is what a comparison against this library's own arithmetic needs. `native-default` is the engine
an ordinary call there gets, which is a policy: its Level 1 is the installed library above the measured
per-operation widths, and a whole dense Level 2 or 3 call goes to that library once the call has enough
arithmetic to pay for reaching it and the routine's documented result is one a library also gives. The
second condition is why a `gemm`, `gemmt` or `syrk` row carrying a multiplier other than one reads as
portable: those three document their result as a multiplier applied to an accumulated sum, and a library may
legally scale an operand instead, so a scaled one of them stays on this library's schedule. Rows on
that arm therefore mix routes, and each says which it took. Its `metadata.txt` section carries the resolved
library file, the library's own version string and the threading evidence the binding read back, so a report
records which binary produced the rows that reached one. A `native-default` row is a default-policy
measurement of that composition and not an exact measurement of either side of it; the `native` arm and the
explicit `<vendor>` arms are where each side is measured on its own.

No whole sparse entry point is bound on either runtime, so every sparse Level 2 and 3 row on
`native-default` names this library's own CSC scheduling exactly as on `native`. What can differ between the
two is the Level 1 kernel a column reaches, which on `native-default` is the library's above the measured
widths and is named as a component where it runs.

Each vendor contributes two explicit arms: `<vendor>` through the Native binding, and `<vendor>-jvm`
through the JVM binding, whose timing includes operand transfer.

Every BLAS invocation runs on one compute thread; there is no thread setting to pass. Operations no vendor
exports — `sum` and everything sparse — are reported unsupported on vendor targets rather than timed through a
substitute.

A Level 1 row is published only where the call's own facts settle which kernel produced the result. A
euclidean norm tries a plain sum of squares and retries through the rescaling loop when that leaves the
normal range, so `nrm2`, `spnrm2` and `spnrm2-indexed` are declined on an arm whose vector kernel would start
them rather than reported under either. A rotation on `native-default` is declined for the same reason: one
run rotated against itself stays portable at every width, since reference `drot` reads an entry back after
storing it, and a route carries no operand identity to tell the two cases apart. Each of those kernels is
measured on the arms where one answer holds, which are the portable `jvm-scalar` and `native` arms and the
explicit `<vendor>` arms.

Built-in dense Level 2/3 rows name `portable-dense/<operation>` and do not resolve a vendor. Where a window of
work is handed to a panel or a Level 1 kernel they name that component too, as
`portable-dense+<component>/<operation>@<group>`, where the group is how many logical columns the backend
recommended handing over at a time. A group is not a lane count.

A matrix product names what it copies and what it computes with. One large enough to be packed reads
`portable-dense+portable-pack/right-panel+portable-pack/left-panel+<tile>/product-block/gemm`, in the order
the schedule reaches them, and a product whose rows leave a tile short names the body that remainder reaches
as well and is published as the composition it is. One too small or too thin to pay for a copy names the
panel it runs on instead, with the grouping the backend recommended, and names a gathered coefficient column
where it makes one. A retained panel is not packed again, so `gemm-packed` names no packing at all and the
mixed entry points name only the side they still copy.

The structured Level 3 routines are the same schedule with their own regions. `gemmt`, `syrk` and `syr2k`
name the packing and tile bodies of the product they are, plus `portable-select/triangle-tile` where a block
straddles the diagonal and only part of its tile reaches the destination; `syr2k` is two such products and
says so. `symm` names `portable-mirror/diagonal-block` for the copy of one diagonal block of its symmetric
operand into a square, and then the windows the strips beside it become. A triangular routine names the
substitution its diagonal blocks reach, as `<backend>/diagonal-solve` or `/diagonal-multiply`, preceded by
`portable-gather/rhs-block` where the backend asked for a block of right-hand sides to be copied adjacent
first, and followed by the windows the products between diagonal blocks became. So a solve over one
right-hand side may name a portable substitution beside a Vector API panel, which is what it runs, and a
route that named one engine for both would be hiding half the call. An order inside a single diagonal block
names no product at all. The right-hand-side grouping is a separate number from the panel grouping the row
carries and is named in the route's reason rather than in `@<group>`.

A triangular or symmetric call whose windows shrink past a
backend's shortest vector window reaches more than one body and is published as the composition it is; a
call whose windows are all empty names no panel and no grouping at all. Built-in sparse
Level 2/3 rows name `portable-csc/<operation>`, and where a unit of work is handed to a Level 1 kernel they
name that component too, as `portable-csc+<component>/<operation>`: the sparse scheduling is this library's own
portable code on every engine, so an arm whose Level 1 kernels are Vector API ones is not thereby running a
vectorised sparse product. The destination scaling a non-unit `beta` performs is a component like any other
and is named. A `direct` row's components all ran; a `composed` row's are the ones the call can reach, because
its columns straddle a kernel's crossover or because the traversal decides per unit whether the kernel is
called at all, and the route says which. A sparse Level 2 or 3 row whose unit of work is a group of
right-hand sides carries that grouping as
`@<group>`, or `@<group>+<tail>` where the last group is shorter, which is the local geometry the backend
recommended and not a lane count. It names the panel body that group reached, as
`<backend>/sparse-rhs-gather`, `/sparse-rhs-scatter` or `/sparse-rhs-mirror`, and a call that
copied its right-hand sides into adjacent order first names that copy as `portable-stage/rhs-source` or
`portable-stage/rhs-destination` — a source is read in, a destination is read in and written back, and a
symmetric product does both. The body named is the one the runs a call actually hands over reached, so a
triangular matrix storing only its diagonal, or a symmetric operand storing only the triangle the call does
not select, names no panel at all. A prepared row reports the schedule the snapshot runs. For a product
against a dense block that is the schedule a one-shot call of the same facts runs, because the snapshot
takes the transpose flag rather than deriving the opposite orientation; for a transposed product against a
second sparse operand it is the untransposed traversal over the derived orientation, and a `firstuse` row
names the preparation and that derivation ahead of it.

A row whose route could not settle what the call runs from the facts that call carries is marked
`+unresolved` before its entry point rather than published as though one schedule had been established.

A dense Level 2 or 3 row on `native-default` whose whole call went to the installed library reads
`host-dense+<vendor>/<symbol>/<operation>`, and one that stayed on this library's schedule reads the
`portable-dense...` forms above, including the vendor Level 1 leaf where a window of work reached one. So the
absence of `host-dense` on a row is not the absence of the library, and the components are what say which
leaves ran. A call that had to copy an operand sharing its destination before the library could take it names
that copy as `host-stage/alias`.

Explicit host rows derive their entry point, resolved library binary, identity, version and threading
evidence from the binding that performs the call. Existing binding route checks retain explicit no-work and
composed-call attribution.

## Cases

[`cases.txt`](cases.txt) lists every workload, one per line, as
`operation+dimensions+fixture[+option=value...][+suite=...]`. Untagged cases belong to `default`;
`+suite=sweep` is opt-in and `+suite=default,sweep` puts one case in both. A workload is listed once, and
membership is not part of a case's identity, so retagging leaves historical comparisons valid.

Sweeps exist for `dot`, `sum`, `asum`, `nrm2`, `iamax`, `axpy`, `scal`, `swap` and `rot`. To add sizes, add
lines with the same fixture and options as the case they extend.

`copy-overlap` copies adjacent overlapping slices of one buffer. `axpy-views` and `swap-views` use disjoint
interleaved views of one buffer, and `spdot-view` reduces a sparse vector against a dense view with stride
two. These include the public vector entry point's alias checks and storage adaptation, so they are
default-policy rows on the selected engine's arm. Their `portable-vector` attribution names the wrapper,
followed by the array copy, scalar sparse view reduction or routed dense Level 1 leaf it actually reaches.
Mutating cases reset the whole shared buffer inside the timed region; dot times only the reduction. Before
timing, each is compared against the explicit scalar engine, including the mutating call's whole buffer.

`spdot-slice`, `spnrm2-slice`, `spaxpy-slice` and `spscatter-slice` use the raw indexed overloads with index
offset three and value offset five, surrounded by invalid indices and NaN padding. These match the array
windows sparse solvers pass from CSC columns and indexed workspaces. They include validation and dispatch
through the platform-selected engine, so they are default-policy rows even when that engine delegates to a
scalar indexed leaf. Their attribution includes `portable-index-validation` and the leaf the route names;
a value-dependent vector norm names its possible scalar rescaling as well. Reductions time arithmetic alone;
mutations reset the dense destination inside the timed region. A separate sequential reduction checks dot
and norm, and the explicit scalar engine checks the complete mutation destination before timing.

`spbuild-triplets` constructs CSC storage from reversed coordinate arrays. `spbuild-duplicates` repeats each
coordinate twice with half its value, so it also measures duplicate reduction and trimming. Both include
the fresh result and all factory scratch in their `construction` timing and retain the complete result.
They are portable construction, measured on the scalar arm as `portable-csc/ofTriplets`. A separate scalar
coordinate sum checks values, stored support and CSC ordering before timing.

`+timing=arithmetic` excludes the per-iteration reset for `scal` and `spgather`; arithmetic scaling uses
alpha = -1. Compare only cases with matching options, since prepared and one-shot timings differ.

`+mode=` says how a prepared sparse operand is accounted for, and the five values measure different logical
work rather than the same work at different speeds:

| Mode | Timed region |
|---|---|
| `oneshot` | The whole call, with no snapshot built at all. |
| `prepared` | Steady-state reuse of a snapshot built before the timed region. |
| `setup` | Building the snapshot, and nothing else. |
| `firstuse` | Building the snapshot and calling it once, which is where a sparse-sparse product's derived orientation is paid for. |
| `amortized` | Building one snapshot and using it `reuse` times, read against that many one-shot calls. |

`spgemv`, `spmm` and `spgemm` carry all five; `spmm-right` has no prepared form and carries `oneshot`
alone, as every other sparse matrix case does. `+transA=T` transposes the sparse operand. For `spgemm` that
is what makes `firstuse` differ from `setup`: a prepared transposed product against a second sparse operand
derives its orientation once, and that derivation is inside the first use and outside the steady one. For
`spgemv` and `spmm` it does not, because those take the flag to the traversal and derive nothing, which the
calibration measured faster than the oriented schedule. `+transB=T` on `spmm` and `spmm-generic` stores the
dense operand the other way round, so the right-hand sides a panel is cut from become adjacent rows rather
than adjacent columns. It is the same result over a different layout, and the two layouts are separate rows
rather than one.

An `amortized` row reports one preparation and all `reuse` uses together: its unit of work is the batch, and
the comparison a break-even needs is that batch against the same number of one-shot calls. Dividing it by
`reuse` is a valid derived per-use cost with the setup spread through it, which is the number to compare
against a single `oneshot` row. What that derived figure is not is a `prepared` row, which is the steady
state with the setup already paid; comparing the two says how much of a prepared call's advantage the
preparation has eaten at that many uses.

`+support=` draws the sparse operand's stored entries from a named distribution at the density the case
already gives, because where they are costs as much as how many there are. `uniform` scatters the same count
over every column, `banded` keeps a column's entries next to the diagonal, `skewed` gives an eighth of the
columns most of the entries, `empty` leaves a quarter of them storing nothing, and `mixed` alternates short
columns with long ones. Only a case whose operand is a sparse matrix takes it.

Before a dense Level 2 or 3 case is timed, its whole destination buffer is compared against an explicit
scalar reference written from the textbook definition, which shares no code with the scheduling it checks.

Before a sparse case is timed, its whole result is compared against an explicit scalar reference computed from
the densified operands, and a fresh CSC result is compared against the support its operands' patterns reach as
well. A prepared case is checked through a snapshot that has not been used yet, so what a transposed call
derives or does not derive on first use is covered rather than assumed. A case that computes the wrong thing
fails the capture instead of publishing a number.

The `panel-multidot`, `panel-columnupdate`, `panel-coupled` and `panel-rankupdate` cases time one raw panel
over the extents the case names, as rows by logical columns. The extents are the case's and never the
backend's, so a tail of three columns is the same requested work on an arm that groups by two and one that
groups by four; what the row records is which body those extents reached and which grouping the backend
asked for. Each is checked against the written-out definition of its panel before it is timed.

The `product-block` cases time one raw product block over the extents the case names, as rows by columns by
depth, on panels the case packs itself through the layout's own published index formula. The extents are the
case's and never the backend's, so the same requested work is timed on an arm whose tile is four rows deep
and one whose tile is eight; what the row records is which bodies those extents reached and the tile shape
the backend cut them with.

`gemm-pack` times packing both operands and nothing else, and reports preparation rather than an arithmetic
kernel, because no product happened. `gemm-packed`, `gemm-packed-left` and `gemm-packed-right` time a product
over operands packed before the timed region, retaining both of them or one. They are four different amounts
of work rather than one row with a flag: the copy a call still makes is the difference between them, and the
packing-only row is what the retained ones have to be read against. Every one of them is checked against the
scalar reference before it is timed.

`spmm-right` is the sparse operand on the right of a dense one through the engine the row is published
under, which is what makes it an exact arm rather than the default-policy `spmm-generic-right` beside it.

The `gemm-generic`, `spmm-generic`, `spmm-generic-right` and `spgemm-generic` cases go through the common
`Matrix` product with its dispatch included, the third of them with the sparse operand on the right of a
dense one. That entry point uses the engine this platform selected rather than one a benchmark names, because a caller holding a
`Matrix` has no engine to pass. They are therefore `default-policy` rows on the arm whose engine is the
selected one, and are declined on every other arm rather than publishing that arm's label over another
engine's work. On a JVM the exact arm admits them only when it is also the selected engine; a default that composes
a host is a separate engine, so those generic calls are admitted only by `jvm-default`. On Kotlin/Native that arm is `native-default`; the exact portable `native` arm
is not the selected engine and declines them. On a Kotlin/Native host with no library installed the two arms
resolve to the same engine, so both admit these rows and both are that host's default policy.

## Direct Gradle runs

CI's `linuxX64SelectedBenchmark` uses a debug executable to check OpenBLAS resolution and invocation without
release optimization. The timed Native benchmark tasks below continue to use release executables.

```bash
./gradlew :koblas-bench:jvmSimdBenchmark -Pbench.suite=sweep -Pbench.operation=dot \
  -Pbench.samples=5 -Pbench.warmups=5 -Pbench.targetMs=200 -Pbench.forks=2
```

Tasks are `jvmScalarBenchmark`, `jvmSimdBenchmark`, `jvmDefaultBenchmark`, `nativeBenchmark`,
`nativeDefaultBenchmark`, `nativeVendorBenchmark` and `jvmVendorBenchmark`; the vendor tasks take
`-Pbench.vendor=NAME`. Native and vendor executables accept `--suite=sweep --operation=dot` directly.
Omitting the suite selects `default`.

## Bench SDK adapter

`lab/bench.json` owns the prepared JVM adapter's build tasks, entry point and
classpath. `lab/native-linux-x64.json` also links the Linux x86_64 executable.
The JVM adapter imports `com.eignex:lab-bench:0.1.0` only from `jvmMain`; Native
continues to use this repository's executable and report formats.

The default dependency resolves [Lab Bench 0.1.0](https://github.com/Eignex/eignex-lab/releases/tag/lab-bench-v0.1.0)
from public GitHub Release assets without a GitHub login or token. Prepare the adapter with:

```sh
./gradlew :koblas-bench:prepareLabBenchmark
```

`-PlabBenchVersion=VERSION` selects a different SDK release. To develop against a
local SDK checkout, run `./gradlew :lab-bench:stageAdapterRepository -PbenchVersion=0.1.0`
in that checkout, then add
`-PlabBenchRepository=file:///absolute/eignex-lab/lab-bench/build/adapter-repository`
to the consumer build. Refresh dependencies when restaging the same version.

A lab submission uses `repository: "Eignex/koblas"` and
`bench: {"manifest":"koblas-bench/lab/bench.json","request":{"parameters":{"targets":"jvm-scalar,jvm-simd","operation":"dot","smoke":"true"}}}`.
Request keys are `targets`, `libraries`, `operation`, `suite`, `cases`, `samples`,
`warmups`, `target-ms`, `forks`, `pass`, `rounds`, `cpu`, `jvm-gc`, `jvm-heap`,
`vendors-only`, `smoke`, `memory-mib`, `timeout-seconds` and `native-executable`.
Each plan freezes one capture with one CPU and the selected memory/time limits.
`rounds` defaults to one; repeated captures rotate target order and assign their own
pass numbers, so `pass` cannot be combined with `rounds` greater than one. `smoke`
and `vendors-only` accept the strings `true` and `false`. Vendor-only requests need
explicit vendor targets; omitted libraries are inferred from those targets. Explicit
libraries must include every selected vendor (`all` uses the harness's platform
selection, which excludes AOCL).

`cases` is a portable path relative to the repository root, for example
`koblas-bench/cases-reductions.txt`. The plan declares it as a source `BenchInput`
so the lab inventories and freezes its contents before dispatch. The adapter resolves
it against the relocated worktree and passes it as one literal argument. A Native
executable is also a declared input, under `koblas-bench/build/`, with a recognized
`linuxX64`, `linuxArm64` or `macosArm64` path component. The selected targets and
executable determine required platform/architecture tags; incompatible combinations
are rejected during planning. The adapter owns `--prepared`, `--retain-incomplete`
and the output destination; these are not request parameters.

JVM targets run in separate processes with the Vector API opened only for
SIMD/default targets. `jvm-gc` accepts `g1`, `serial` or `zgc`; `jvm-heap` fixes both
initial and maximum heaps and accepts a positive size ending in `m` or `g`. Both
controls require a JVM target. The per-JVM heap budget is
`(memory-mib - 768) / 2` MiB to allow the launcher and its measured JMH fork to
coexist, with 768 MiB reserved for the adapter and other process overhead. An explicit
heap must fit this budget. Memory defaults to 2048 MiB with a minimum of 1024 MiB;
timeout defaults to 3600 seconds and must be positive. `cpu` accepts `auto` or a
logical CPU number and adds a Linux requirement. `auto` chooses from the worker's
allowed affinity mask; a numeric CPU must be allowed on that worker. Pinning does
not reserve the CPU.

For example, the request parameters below capture two JVM arms over two rounds
with a frozen custom case file and fixed collector/heap settings:

```json
{
  "targets": "jvm-scalar,jvm-simd",
  "cases": "koblas-bench/cases-reductions.txt",
  "operation": "dot",
  "rounds": "2",
  "cpu": "auto",
  "jvm-gc": "serial",
  "jvm-heap": "256m",
  "smoke": "true"
}
```

The result's `captured-rows` counts nonblank data rows in the selected targets' CSVs,
from the report root for one round or `pass-1`, `pass-2`, ... for repeated captures.
`captured-csv-files` counts those files and `completed-rounds` counts round reports
marked complete. Summary files and nested sidecars are retained but do not add rows
to these metrics. Repeated rows remain separate observations; the adapter does not
aggregate timing values. Success requires a zero exit status, complete outer and
round metadata, and at least one captured row. Failed captures report the rows and
completed rounds available so far, including partial output from the failed round.

`capture-report.sh --targets TARGET,...` selects exact capture arms.
`--prepared` uses `KOBLAS_BENCH_JAVA` and `KOBLAS_BENCH_CLASSPATH` without Gradle;
Native arms require `--native-executable`. The adapter uses `--retain-incomplete`,
which requires an absent report destination and retains incomplete metadata,
partial CSVs and target logs on failure. The lab keeps that child report tree and
a separate `capture.log`. Ordinary captures retain their atomic replacement behavior.
Smoke results verify integration and are not performance evidence. Linux arm64
and macOS Native binaries require their corresponding preparation tasks and
hardware validation before using those targets.

### Allocation measurements

Set `KOBLAS_BENCH_PROFILE_GC=true` (or `profile-gc=true` in the vendor comparison adapter) to retain
JMH allocation measurements alongside timing in the target logs. Raw product-block cases separate
kernel allocation from the temporary panel arrays used by whole products without a workspace.
The eight-column prototype measured at revision `a23a3736` was removed after those measurements
exposed extra allocation; its pinned source retains the experimental comparison target.

### Thin GEMM and workspace reuse

`gemm` accepts `workspace=retained` to reuse one ordinary-array `Workspace` across setup and measured
calls. Omitting the option keeps the existing call without a workspace. Vendor arms execute the same
arithmetic through their binding, which has no workspace parameter and still includes native transfers.
Compare both owned cases before selecting a native crossover: removing repeated panel allocation can
change which implementation wins. `lab/cases-thin-gemm-comparison.txt` includes depth, shape and
transpose controls for that decision. JVM-only vendor captures skip linking an unused native executable.
