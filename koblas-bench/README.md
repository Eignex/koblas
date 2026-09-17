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

An ARM64 instance captures OpenBLAS and the Koblas arms only. oneMKL has no ARM64 build, and Arm Performance
Libraries is not in the image because its download requires accepting a licence, so Level 2 and 3 have no
vendor to reach and report as unsupported. Installing ArmPL on the instance and capturing without the
container is the way to cover that until the image can carry it.

## Options

| Option | Effect |
|---|---|
| `--libraries openblas,accelerate,onemkl\|all` | Which vendors to run. `all` is OpenBLAS and oneMKL on Linux, OpenBLAS and Accelerate on macOS. |
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
