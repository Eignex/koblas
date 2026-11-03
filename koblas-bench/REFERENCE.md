# Benchmark reference

Start with the [README](README.md) for commands and target names. This page covers the settings and
timing boundaries that matter when comparing results. Full options are in `capture-report.sh --help`.

## Capture settings

Use `--targets` to select implementations and `--cases` to select a workload file. `--operation` filters
that file; `--suite sweep` selects its opt-in sizes and requires an operation. Vendor targets need a
compatible installed library; the [engine guide](../BACKENDS.md) covers setup.

`--warmups` discards initial repetitions, `--samples` sets measured repetitions, and `--target-ms` sets
the time budget per repetition. `--forks` controls JVM forks. `--rounds` repeats the capture with rotating
target order; give repeated captures a new output directory. Smoke runs only check integration.

Results default to `build/benchmarks/<hardware-sha256>/`, or the directory passed to `--output`.
Each target gets a CSV and log, with hardware, runtime, source and library details in `metadata.txt`.
Repeated captures retain each round under `pass-1`, `pass-2`, and so on. Compare the median of each
case's round medians, retain the spread, and check unchanged controls.

For local Linux captures, `--cpu spread` picks a lightly loaded allowed core; `--cpu auto` picks the
first allowed core. Pinning does not reserve it. Shared hosts are supported, but unexplained changes
in controls prevent attributing a timing difference to the code.

Use `--jvm-gc g1|serial|zgc` and `--jvm-heap 1g` to fix collector and heap settings. Direct Gradle runs
also accept `-Pbench.javaVersion=27` and `-Pbench.expectedJavaHome=/path/to/jdk` to select the measured
runtime. Set `KOBLAS_BENCH_PROFILE_GC=true` to retain JMH allocation measurements in target logs.
The JVM harness uses a pinned JMH build; reports retain its version and revision.

Windows captures run in Git Bash with JDK 25 and matching vendor DLLs on PATH; Linux CPU pinning is
unavailable there. Windows Arm64 uses JVM targets. Native Linux ARM64 captures need a cross-built
executable: build with `:koblas-bench:linkReleaseExecutableLinuxArm64` on x64, then pass the binary
with `--native-executable`. Without it, ARM64 Linux captures cover JVM targets only. Compile checks
do not establish performance on the target host.

## Cases

[Cases](cases.txt) use `operation+dimensions+fixture[+option=value...]`. Untagged cases belong to
`default`; `+suite=sweep` is opt-in. Compare identical case options: shape, transpose, sparse support
and reset policy all change the work being timed. Sparse matrix cases carry `mode=oneshot`, the only
timing mode, which times the whole call.

Other timing distinctions to keep in mind:

- Ordinary mutating cases commonly include destination reset. `timing=arithmetic` excludes it where supported.
- `axpy-unit` accumulates with alpha one without a timed reset; ordinary `axpy` uses a nonunit multiplier and includes reset.
- `ger-shrinking` and `ger-window-shrinking` report a batch of `n - 1` updates, excluding matrix reset and factorization work.
- GEMM's `workspace=retained` reuses scratch; `layout=borrowed` uses offsets and padded storage. All engine arms share these addressed fixtures.
- Sparse construction cases include allocation; view and raw indexed cases include their public validation and dispatch.

Setup validates results against scalar references before measurement. Kernel timings do not establish
an end-to-end solver speedup. For exact boundaries, see the implementations in
[Dense.kt](src/commonMain/kotlin/com/eignex/koblas/bench/Dense.kt),
[Sparse.kt](src/commonMain/kotlin/com/eignex/koblas/bench/Sparse.kt), and
[ShrinkingRankUpdate.kt](src/commonMain/kotlin/com/eignex/koblas/bench/ShrinkingRankUpdate.kt).

An engine row's `comparison_kind` is `engine` and its `actual_kernel` is the selected engine's name and
the operation: a selection, not the body a given size reached. Exact arms pin an engine through
`BuiltinEngines`, which is what makes them exact. Default engines can mix Kotlin and native calls. A default
engine that loaded a native library names it, with its path, version and thread evidence, in the
target's `implementation` line in `metadata.txt`. The `provider` column records the Level 1 or whole-call
provider selected by the same internal dispatcher the call uses. `scheduled` denotes shared scheduling
with owned kernels, including portable CSC traversal; it does not claim a single arithmetic body. SIMD
targets can take scalar fallbacks below their crossovers, and sparse scheduling remains Kotlin even when
arithmetic uses SIMD. Vendor rows name the implementation supplied by the binding. JVM vendor timings include
array transfers.

Dense vector cases accept `stride=2`, `stride=-1` and `stride=-2`; omitted stride means contiguous owned
arrays. `copy` measures the same checked engine operation as the other arms. Routing keys record
`operation:rows:columns:depth:borrowed:level:contiguous`; indexed vector keys use stored counts.

## Lab captures

Use the authenticated API at lab.eignex.com through the maintained lab client. The standard manifest is
[lab/bench.json](lab/bench.json); native manifests also prepare their platform's executable.
Build the JVM adapter with `./gradlew :koblas-bench:prepareLabBenchmark`.

A request identifies `repository: "Eignex/koblas"` and supplies:

```json
{
  "manifest": "koblas-bench/lab/bench.json",
  "request": {
    "parameters": {
      "targets": "jvm-scalar,jvm-simd",
      "cases": "koblas-bench/cases-reductions.txt",
      "rounds": "3"
    }
  }
}
```

Pass this object as `bench`. Each selected workload becomes one paired case; paths are relative to the repository.
The optional `profile` parameter names a source-relative configuration file, frozen with role `profile`
and exported as `KOBLAS_BENCH_ENGINE_PROFILE` for default-engine arms. One adapter handles owned and vendor
captures, with `vendor`, `platform` and `arch` selecting compatible placement. It uses 2 GiB per case,
a 512 MiB JVM heap and serial GC; `samples`, `warmups`, `target-ms`, `forks` and `rounds` set the timing
protocol. `jvm-options`, `profile-gc` and `smoke` are explicit options. Direct captures retain the CLI's
other controls. The adapter owns the output directory and retains logs and partial results on failure.
Windows workers need Git Bash or an absolute KOBLAS_BASH.
Use automatic placement. Keep all arms of a comparison row on one machine in one worker attempt;
replacement attempts rerun the complete unfinished row. Other rows may use different compatible machines.
See [LabAdapter.kt](src/jvmMain/kotlin/com/eignex/koblas/bench/LabAdapter.kt) for accepted parameters and validation.

The [calibration manifest](lab/calibration.json) prepares the same generated grids as local tuning under
`koblas-bench/build/calibration/`. Set `cases` to `training.txt` or `holdout.txt` for JVM, or
`native-training.txt` or `native-holdout.txt` for Native. `native-tail.txt` supplements identical measured
sources; changing measured code requires a fresh capture of the target.
Use [the macOS manifest](lab/calibration-macos.json) for Accelerate so preparation builds on a compatible
Mac. The [Linux ARM64 manifest](lab/calibration-linux-arm64.json) declares x64 preparation and ARM64
build outputs, so the lab hands the cross-built executable to compatible ARM workers.
Native executables are frozen inputs built once during preparation; measured rows never invoke Gradle.
Each workload has its own paired case, uses the lab's normal CPU accounting and 2 GiB, and runs without
CPU affinity pinning. Placement chooses compatible hardware automatically. `native-timing.txt` is a small
subset for comparing timing protocols before changing measurement settings.
The exact `jvm-simd` arm measures every supplied owned contribution from its intrinsic minimum length.
Generation requires Python 3 (`KOBLAS_PYTHON` can name its executable).
OneMKL comparisons may also include OpenBLAS arms, keeping vendor-precedence measurements in the same
worker attempt. Generated grids are declared inputs; they are not checked into source control.

### Score lab comparisons

The scripts under [lab](lab) read fetched job directories, whatever their case layout, and need only Bash and awk:

```bash
# Ratio of jvm-simd to openblas per case, paired within each pass and weighted equally over CPU models.
koblas-bench/lab/compare.sh results/run-1 jvm-simd openblas >run-1.txt
# Several jobs: pool their cases in one directory so the weighting spans all of them.
mkdir -p pooled && cp -r results/run-1/cases pooled/run-1 && cp -r results/run-2/cases pooled/run-2
koblas-bench/lab/compare.sh pooled jvm-simd openblas >pooled.txt
# Win or lose per routine and size band.
koblas-bench/lab/score.sh openblas pooled.txt
```

The lab places each job automatically, and one pool can mix CPU models; the x86 AVX-512 pool mixes Ice Lake
and Sapphire Rapids, whose ratios for one commit differ by up to a third in a cell. A pass is one machine, so
`compare.sh` forms each ratio within a pass, takes the median per CPU model from the pass's `metadata.txt`,
and weighs the models equally, so a different mix of machines cannot move the result. `median.sh` combines
finished comparisons without that weighting; use it only for runs on one CPU model. When comparing jobs
over time, compare like models or note where the mix changed.

To compare a variant, put it and its base in one job as `configs` arms with `repeats` of at least 3, so that
each row runs every arm on one machine, and pass the arm's label as the fourth argument to `compare.sh`. It
then reads only rows in which every arm finished. `score.sh` documents its size bands and the Accelerate
coprocessor exemption.

### Prepared vendor comparisons

The same calibration manifests prepare vendor comparisons: Linux x64 pins the oneMKL reference,
macOS supplies Accelerate, and Linux ARM64 cross-builds on x64 because Kotlin/Native has no ARM compiler host.
Use the ARM manifest with OpenBLAS and `arch=arm64`; execution rows select compatible ARM workers automatically.
Timed workers use these frozen inputs without invoking
Gradle or downloading vendor packages. JVM-only arms need no Native executable input.

Each workload is a separate paired lab case. All selected arms run on the same compatible machine in one
attempt, with automatic placement and no CPU affinity. The SDK's default CPU accounting and 2 GiB memory
allowance cover measurement rather than reserving compiler resources throughout the run. Completed rows
are visible individually, and a lost worker repeats only the unfinished workload.
