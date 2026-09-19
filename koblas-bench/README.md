# koblas-bench

CPU benchmarks for Koblas, OpenBLAS, Accelerate and oneMKL. Requires JDK 25. `capture-report.sh` is the only
reporting script.

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

Builds an image with a pinned JDK, OpenBLAS and oneMKL, mounts the repository, and captures inside it, so a
report from a cloud host is comparable with one from a laptop. Takes `openblas`, `onemkl` or `all` before the
capture options; `all` needs an x86-64 host, since oneMKL ships no ARM64 build. Requires Docker, and the
report lands in the repository as usual because the working tree is mounted rather than copied.

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

An ARM64 Linux capture is the JVM arms and `armpl-jvm` alone, and `native_capable=false` in `metadata.txt`
marks a report that ran under that limit. The Kotlin/Native compiler ships no linux-aarch64 host, so the
executable cannot be built on the machine being measured.

That is a limit on building in place, not on measuring. The compiler cross-compiles to `linuxArm64` from an
x86-64 host, so the executable can be built elsewhere and carried to the instance. Nothing here does that
yet, which is why the ARM rows of the ladder have no `native` or plain `armpl` numbers at all, and why the
Level 1 kernels Koblas ships for `linuxArm64` are the only production code in the library that no benchmark
has ever timed.

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

Three gaps, none of them a lane count.

The ARM rungs have no `native` or plain `armpl` numbers, for the build reason given above. That leaves the
`linuxArm64` Level 1 kernels as the only production code in the library nothing has ever timed, and it is a
matter of carrying a cross-compiled executable to the instance rather than of anything being unmeasurable.

The AMD rung is Zen 2, which stops at AVX2, so every AVX-512 number in the fleet is Intel's. AMD implements
that width differently enough that it is not the same rung read twice, and `c7a.2xlarge` is where Zen 4 has
it. The AMD host also selects its vendor differently: production tries AOCL first there, and the container
builds no AOCL stage, so the library that a Koblas process on an AMD machine reaches before any other is one
no capture has ever measured.

Accelerate and macOS are absent, and the Linux fleet cannot stand in. A dedicated Apple silicon machine is
the way to cover it, physical or the EC2 kind that rents at a 24-hour minimum. It is the one vendor Koblas
selects with no fallback behind it.

## Options

| Option | Effect |
|---|---|
| `--libraries openblas,accelerate,onemkl,armpl\|all` | Which vendors to run. `all` is OpenBLAS with oneMKL on x86-64 Linux, ArmPL on ARM64 Linux and Accelerate on macOS. |
| `--suite default\|sweep` | Case suite. `sweep` requires `--operation`. |
| `--operation NAME\|all` | Intersects the suite with one kernel. |
| `--samples N`, `--warmups N` | Measured and discarded repetitions per case. |
| `--target-ms N` | Time budget per repetition. |
| `--forks N` | JVM forks. Native and vendor executables ignore it. |
| `--smoke` | First three selected cases at one short sample. |
| `--vendors-only` | Skips the Koblas targets. |
| `--output DIR` | Overrides the report directory. |
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

Koblas contributes independent `jvm-scalar`, `jvm-simd` and `native` arms. In S1 both JVM built-in arms use the
portable scalar Level 2/3 component and report that component rather than implying SIMD or vendor execution;
later panel/tile stages replace eligible components. Each vendor contributes two explicit arms: `<vendor>`
through the Native binding, and `<vendor>-jvm` through the JVM binding, whose timing includes operand transfer.

Every BLAS invocation runs on one compute thread; there is no thread setting to pass. Operations no vendor
exports — `sum` and everything sparse — are reported unsupported on vendor targets rather than timed through a
substitute.

Built-in Level 2/3 rows name `portable-scalar/<operation>` in S1 and do not resolve a vendor. Explicit host rows
derive their entry point, resolved library binary, identity, version and threading evidence from the binding
that performs the call. Existing binding route checks retain explicit no-work and composed-call attribution.

## Cases

[`cases.txt`](cases.txt) lists every workload, one per line, as
`operation+dimensions+fixture[+option=value...][+suite=...]`. Untagged cases belong to `default`;
`+suite=sweep` is opt-in and `+suite=default,sweep` puts one case in both. A workload is listed once, and
membership is not part of a case's identity, so retagging leaves historical comparisons valid.

Sweeps exist for `dot`, `sum`, `asum`, `nrm2`, `iamax`, `axpy`, `scal`, `swap` and `rot`. To add sizes, add
lines with the same fixture and options as the case they extend.

`+timing=arithmetic` excludes the per-iteration reset for `scal` and `spgather`; arithmetic scaling uses
alpha = -1. Compare only cases with matching options, since prepared and one-shot timings differ.

## Direct Gradle runs

```bash
./gradlew :koblas-bench:jvmSimdBenchmark -Pbench.suite=sweep -Pbench.operation=dot \
  -Pbench.samples=5 -Pbench.warmups=5 -Pbench.targetMs=200 -Pbench.forks=2
```

Tasks are `jvmScalarBenchmark`, `jvmSimdBenchmark`, `nativeBenchmark`, `nativeVendorBenchmark` and
`jvmVendorBenchmark`; the vendor tasks take `-Pbench.vendor=NAME`. Native and vendor executables accept
`--suite=sweep --operation=dot` directly. Omitting the suite selects `default`.
