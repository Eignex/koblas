# koblas-bench

CPU benchmarks for Koblas and the vendor libraries it selects: OpenBLAS, oneMKL, AOCL, ArmPL and Accelerate.
Requires JDK 25. `capture-report.sh` is the only reporting script.

## Running a capture

```bash
# Full capture: JVM scalar and SIMD, Native, and the platform's vendors.
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

Koblas contributes independent `jvm-scalar`, `jvm-simd` and `native` arms. Both JVM built-in arms use the
portable scalar dense Level 2/3 component and the portable CSC sparse one, and report those components rather
than implying SIMD or vendor execution; later panel/tile stages replace eligible components. Each vendor contributes two explicit arms: `<vendor>`
through the Native binding, and `<vendor>-jvm` through the JVM binding, whose timing includes operand transfer.

Every BLAS invocation runs on one compute thread; there is no thread setting to pass. Operations no vendor
exports — `sum` and everything sparse — are reported unsupported on vendor targets rather than timed through a
substitute.

Built-in dense Level 2/3 rows name `portable-scalar/<operation>` and do not resolve a vendor. Built-in sparse
Level 2/3 rows name `portable-csc/<operation>`, and where a unit of work is handed to a Level 1 kernel they
name that component too, as `portable-csc+<component>/<operation>`: the sparse scheduling is this library's own
portable code on every engine, so an arm whose Level 1 kernels are Vector API ones is not thereby running a
vectorised sparse product. The destination scaling a non-unit `beta` performs is a component like any other
and is named. A `direct` row's components all ran; a `composed` row's are the ones the call can reach, because
its columns straddle a kernel's crossover or because the traversal decides per unit whether the kernel is
called at all, and the route says which. Explicit host rows derive their entry point, resolved library binary, identity, version and threading
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

`+mode=` says how a prepared sparse operand is accounted for, and the four values measure different logical
work rather than the same work at different speeds:

| Mode | Timed region |
|---|---|
| `oneshot` | The whole call, with no snapshot built at all. |
| `prepared` | Steady-state reuse of a snapshot built before the timed region. |
| `setup` | Building the snapshot, and nothing else. |
| `firstuse` | Building the snapshot and calling it once, which is where a derived orientation is paid for. |

`spgemv`, `spmm` and `spgemm` carry all four. Every other sparse matrix case is `mode=oneshot`, because it has
no prepared form. `+transA=T` transposes the sparse operand, which is what makes `firstuse` differ from
`setup`: a prepared transposed product derives its orientation once, and that derivation is inside the first
use and outside the steady one.

Before a sparse case is timed, its whole result is compared against an explicit scalar reference computed from
the densified operands, and a fresh CSC result is compared against the support its operands' patterns reach as
well. A prepared case is checked through a snapshot that has not been used yet, so the orientation a
transposed call derives on first use is covered rather than assumed. A case that computes the wrong thing
fails the capture instead of publishing a number.

The `spmm-generic`, `spmm-generic-right` and `spgemm-generic` cases go through the common `Matrix` product with
its dispatch included, the second of them with the sparse operand on the right of a dense one. That entry
point uses the engine this platform selected rather than one a benchmark names, because a caller holding a
`Matrix` has no engine to pass. They are therefore `default-policy` rows on the arm whose engine is the
selected one, and are declined on every other arm rather than publishing that arm's label over another
engine's work. On a Kotlin/Native host with an installed library the selected engine is not the `native` arm's
scalar one, so these cases are declined there too.

## Direct Gradle runs

```bash
./gradlew :koblas-bench:jvmSimdBenchmark -Pbench.suite=sweep -Pbench.operation=dot \
  -Pbench.samples=5 -Pbench.warmups=5 -Pbench.targetMs=200 -Pbench.forks=2
```

Tasks are `jvmScalarBenchmark`, `jvmSimdBenchmark`, `nativeBenchmark`, `nativeVendorBenchmark` and
`jvmVendorBenchmark`; the vendor tasks take `-Pbench.vendor=NAME`. Native and vendor executables accept
`--suite=sweep --operation=dot` directly. Omitting the suite selects `default`.
