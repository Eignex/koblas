# koblas-bench

CPU benchmarks for Koblas and the vendor libraries it selects: OpenBLAS, oneMKL, AOCL, ArmPL and Accelerate.
Requires JDK 25. `capture-report.sh` is the only reporting script.

## Running a capture

```bash
# Full capture: the three JVM arms, Native, and the platform's vendors.
koblas-bench/capture-report.sh --samples 5 --warmups 5 --target-ms 200 --forks 2

# One operation across its opt-in sizes.
koblas-bench/capture-report.sh --suite sweep --operation dot --samples 5 --warmups 5 --target-ms 200 --forks 2

# First three selected cases, one sample, no warmup.
koblas-bench/capture-report.sh --smoke

# Vendors only.
koblas-bench/capture-report.sh --vendors-only --libraries openblas,accelerate
```

On a hybrid CPU, pin the run: `taskset -c 2,4,6,8 koblas-bench/capture-report.sh …`. Otherwise it wanders
between core types and the timings move with the scheduler.

## Capturing in the reference container

```bash
koblas-bench/reference-container.sh all --samples 5 --warmups 5 --target-ms 200 --forks 2
```

Builds an image with a pinned JDK, OpenBLAS and oneMKL, mounts the repository, and captures inside it,
so a report from a cloud host is comparable with one from a laptop. Takes `openblas`, `onemkl`, `aocl` or
`all` before the capture options; `all` needs an x86-64 host, since oneMKL ships no ARM64 build. Requires
Docker, and the report lands in the repository as usual because the working tree is mounted rather than
copied.

On an AMD host use the `aocl` target, which adds the library `Vendor.select` puts ahead of oneMKL there:

```bash
koblas-bench/reference-container.sh aocl --samples 5 --warmups 5 --target-ms 200 --forks 2
```

That stage downloads AMD Optimizing CPU Libraries, which means accepting AMD's licence for it, and is why it
is a separate target rather than part of `all`. The licence covers using it on a machine you are running it
on; it does not cover redistributing it, so an image built from that stage must not be published. The same is
true of `armpl` below, and it is why `all` names only the libraries an image may carry.

### On a cloud instance

Install Docker, clone the repository, and run the command above. Reports are keyed by a hash of the hardware,
so different instance types write to different directories and never overwrite each other.

Take an instance that owns its cores. A burstable type throttles partway through a capture and a shared one
gives the cores away, and either turns up as a slower arm rather than as an error. Sizes at or above the
point where an instance holds a whole socket are what produce repeatable numbers, and pinning still applies:
a vCPU is usually one hyperthread, so pin to every other one.

On ARM64 use the `armpl` target rather than `all`, since oneMKL has no ARM64 build:

```bash
koblas-bench/reference-container.sh armpl --samples 5 --warmups 5 --target-ms 200 --forks 2
```

That stage downloads Arm Performance Libraries, which means accepting Arm's licence for it, and is why it is a
separate target rather than part of `all`. The licence covers installing it on a machine you are using; it
does not cover redistributing it, so an image built from that stage must not be published. ArmPL is worth the
step because it is the tuned library Koblas prefers on ARM64; OpenBLAS is what selection falls back to when
none is installed, so a capture without ArmPL measures only the last resort.

The Kotlin/Native compiler ships no linux-aarch64 host, so an ARM64 machine cannot link the executable it
would time. It can still run one. Build it on an x86-64 host, where the compiler cross-compiles to the
target, carry it over, and name it:

```bash
# On an x86-64 machine, in a clone of the repository.
./gradlew :koblas-bench:linkReleaseExecutableLinuxArm64
scp koblas-bench/build/bin/linuxArm64/releaseExecutable/koblas-bench.kexe instance:/home/ubuntu/

# On the instance.
koblas-bench/capture-report.sh --native-executable /home/ubuntu/koblas-bench.kexe \
  --samples 5 --warmups 5 --target-ms 200 --forks 2
```

Without it the capture is the JVM arms and `armpl-jvm` alone, which `native_capable=false` in `metadata.txt`
records. `native_executable` there says which binary produced the native rows, since one built on another
machine is not the same evidence as one linked where it ran.

Nothing about this is automatic, so an ARM capture that does not pass the flag silently covers less. The
`linuxArm64` Level 1 kernels are shipped production code, and the plain `armpl` arm is the binding a Kotlin/
Native consumer on that hardware actually uses.

### Which hosts to capture

The point of a fleet is the SIMD ladder, so each instance family is here for the vector width and generation
it is the cheapest way to reach, and dropping one leaves a rung unmeasured rather than saving a duplicate.

| Instance | Microarchitecture | Widest SIMD | Double lanes |
|---|---|---|---|
| `m2.xlarge` | Xeon E5-2665 Sandy Bridge | AVX, no FMA3 | 2 |
| `c3.2xlarge` | Xeon E5-2680 v2 Ivy Bridge | AVX, no FMA3 | 2 |
| `c4.2xlarge` | Xeon E5-2666 v3 Haswell | AVX2 with FMA3 | 4 |
| `c5a.2xlarge` | AMD EPYC 7R32 Zen 2 | AVX2 with FMA3 | 4 |
| `c7i.2xlarge` | Xeon Platinum 8488C Sapphire Rapids | AVX-512 | 8 |
| `c6g.2xlarge` | Graviton2 Neoverse-N1 | NEON | 2 |
| `c7g.2xlarge` | Graviton3 Neoverse-V1 | SVE 256-bit | 4 |
| `c8g.2xlarge` | Graviton4 Neoverse-V2 | SVE 128-bit | 2 |

Lane counts are what `SPECIES_PREFERRED` resolved to on JDK 25, read back from each capture rather than
derived from the instruction set: the AVX hosts get 128-bit vectors because HotSpot caps its vector size
where only the first generation is present, and Graviton4's SVE is narrower than Graviton3's although it is
the later part. Every rung therefore has to be measured rather than predicted, which is the reason to keep
the list.

The pre-FMA rungs earn their place even though nobody buys those instances now: a Vector API operation with
no instruction behind it falls back to a software implementation per lane rather than refusing, and only a
host without the instruction shows that.

Between them these cover every property the SIMD kernels branch on: each of the three lane counts, a host on
each side of the fused multiply-add, both 256-bit and 512-bit x86 for the indexed loads and stores that are
eligible at one width and not the other, and both instruction set families.

### Not yet covered

Two gaps, neither of them a lane count. Both need a machine rather than a change here.

The AMD rung is Zen 2, which stops at AVX2, so every AVX-512 number in the fleet is Intel's. AMD implements
that width differently enough that it is not the same rung read twice, and `c7a.2xlarge` is where Zen 4 has
it. That instance matters more than a missing width would on its own, because AMD is also where `Vendor.select`
puts AOCL ahead of everything else, so it is the only host whose first-choice vendor is the one the `aocl`
arm times.

Accelerate and macOS are absent, and the Linux fleet cannot stand in. A dedicated Apple silicon machine is
the way to cover it, physical or the EC2 kind that rents at a 24-hour minimum. It is the one vendor Koblas
selects with no fallback behind it.

## Options

| Option | Effect |
|---|---|
| `--libraries openblas,accelerate,onemkl,aocl,armpl\|all` | Which vendors to run. `all` is OpenBLAS with oneMKL on x86-64 Linux and Accelerate on macOS; AOCL and ArmPL are named explicitly, since their licences keep them out of `all`. |
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
someone else can rent or buy, and cores it does not share. The cloud fleet above is how most rungs are
reached, and a dedicated physical machine qualifies on the same terms, which is the only way to reach
Accelerate.

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
`jvm-default` is what an ordinary call gets, which is a policy rather than an exact arm: today that is
`jvm-simd`'s Level 1 with the portable dense panels, the portable product tile and the portable diagonal
substitution, because the Level 2 panels and the Level 3 tiles and substitutions are measured but not yet
activated. Its rows are where the generic entry points are timed, since those
use the selected engine and have none to be told. A comparison between `jvm-simd` and `jvm-default` is
therefore a comparison of the matrix arithmetic alone, since the two share everything else.

`native` and `native-default` are the same split on Kotlin/Native and are two different engines rather than
one named twice. `native` is the exact portable arm: common Kotlin at every level, resolving no library at
all, which is what a comparison against this library's own arithmetic needs. `native-default` is the engine
an ordinary call there gets, which is a policy: its Level 1 is the installed library above the measured
per-operation widths, and a whole dense Level 2 or 3 call goes to that library once the call has enough
arithmetic to pay for reaching it and the routine's documented result is one a library also gives. Rows on
that arm therefore mix routes, and each says which it took. Its `metadata.txt` section carries the resolved
library file, the library's own version string and the threading evidence the binding read back, so a report
records which binary produced the rows that reached one. A `native-default` row is a default-policy
measurement of that composition and not an exact measurement of either side of it; the `native` arm and the
explicit `<vendor>` arms are where each side is measured on its own.

Sparse has no bound host entry point at all on either runtime. Every sparse Level 2 and 3 row on
`native-default` is this library's own CSC scheduling, exactly as on `native`, and only the Level 1 kernel a
column reaches can be the library's. No supported vendor's sparse API is bound here, so sparse acceleration
is not something a Kotlin/Native user gets from installing one.

Each vendor contributes two explicit arms: `<vendor>` through the Native binding, and `<vendor>-jvm`
through the JVM binding, whose timing includes operand transfer.

Every BLAS invocation runs on one compute thread; there is no thread setting to pass. Operations no vendor
exports — `sum` and everything sparse — are reported unsupported on vendor targets rather than timed through a
substitute.

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
not select, names no panel at all. A prepared row reports the schedule the snapshot runs rather than the
one-shot schedule: a transposed product against a reused orientation takes the untransposed traversal over
it, and a `firstuse` row names the preparation and that derivation ahead of it.

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

`+timing=arithmetic` excludes the per-iteration reset for `scal` and `spgather`; arithmetic scaling uses
alpha = -1. Compare only cases with matching options, since prepared and one-shot timings differ.

`+mode=` says how a prepared sparse operand is accounted for, and the five values measure different logical
work rather than the same work at different speeds:

| Mode | Timed region |
|---|---|
| `oneshot` | The whole call, with no snapshot built at all. |
| `prepared` | Steady-state reuse of a snapshot built before the timed region. |
| `setup` | Building the snapshot, and nothing else. |
| `firstuse` | Building the snapshot and calling it once, which is where a derived orientation is paid for. |
| `amortized` | Building one snapshot and using it `reuse` times, read against that many one-shot calls. |

`spgemv`, `spmm` and `spgemm` carry all five; `spmm-right` has no prepared form and carries `oneshot`
alone, as every other sparse matrix case does. `+transA=T` transposes the sparse operand, which is what
makes `firstuse` differ from `setup`: a prepared transposed product derives its orientation once, and that
derivation is inside the first use and outside the steady one.

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
well. A prepared case is checked through a snapshot that has not been used yet, so the orientation a
transposed call derives on first use is covered rather than assumed. A case that computes the wrong thing
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
engine's work. On Kotlin/Native that arm is `native-default`; the exact portable `native` arm is not the
selected engine and declines them.

## Direct Gradle runs

```bash
./gradlew :koblas-bench:jvmSimdBenchmark -Pbench.suite=sweep -Pbench.operation=dot \
  -Pbench.samples=5 -Pbench.warmups=5 -Pbench.targetMs=200 -Pbench.forks=2
```

Tasks are `jvmScalarBenchmark`, `jvmSimdBenchmark`, `jvmDefaultBenchmark`, `nativeBenchmark`,
`nativeDefaultBenchmark`, `nativeVendorBenchmark` and `jvmVendorBenchmark`; the vendor tasks take
`-Pbench.vendor=NAME`. Native and vendor executables accept `--suite=sweep --operation=dot` directly.
Omitting the suite selects `default`.
