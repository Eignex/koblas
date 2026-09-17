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

An ARM64 Linux capture is the JVM arms and `armpl-jvm` alone. The Kotlin/Native compiler ships no
linux-aarch64 host, so the native executable cannot be built there at any version, which leaves `native` and
the plain vendor targets unreachable rather than merely slow. `native_capable=false` in `metadata.txt` marks a
report that ran under that limit.

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

A report holds one CSV per target plus `metadata.txt`. The CSV has one row per case with sample and fork
counts and median, minimum and maximum ns/op, along with the kernel that actually ran. Run settings, hardware,
source revision and the resolved library files are in `metadata.txt` only.

## Targets

Koblas contributes `jvm-scalar`, `jvm-simd` and `native`. Each vendor contributes two: `<vendor>` through the
Native binding, and `<vendor>-jvm` through the JVM binding, whose timing includes copying the operands into
native memory.

Every BLAS invocation runs on one compute thread; there is no thread setting to pass. Operations no vendor
exports — `sum` and everything sparse — are reported unsupported on vendor targets rather than timed through a
substitute.

The Koblas targets carry their own Level 1 kernels but hand Level 2 and 3 to whichever library production
selection resolved, and that order ends in OpenBLAS. On a host with no tuned library installed, their Level 2
and 3 rows are therefore the same library the `openblas` target times, reached through the same binding, and
comparing the two answers nothing. `metadata.txt` names the resolved file per target, which is what says
whether that happened.

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
