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
- GEMM's `workspace=retained` reuses scratch; `layout=borrowed` uses offsets and padded storage. Explicit vendor arms decline borrowed cases.
- Sparse construction cases include allocation; view and raw indexed cases include their public validation and dispatch.

Setup validates results against scalar references before measurement. Kernel timings do not establish
an end-to-end solver speedup. For exact boundaries, see the implementations in
[Dense.kt](src/commonMain/kotlin/com/eignex/koblas/bench/Dense.kt),
[Sparse.kt](src/commonMain/kotlin/com/eignex/koblas/bench/Sparse.kt), and
[ShrinkingRankUpdate.kt](src/commonMain/kotlin/com/eignex/koblas/bench/ShrinkingRankUpdate.kt).

An engine row's `comparison_kind` is `engine` and its `actual_kernel` is the selected engine's name and
the operation: a selection, not the body a given size reached. Exact arms pin an engine through
`BuiltinEngines`, which is what makes them exact. Default engines can mix Kotlin and native calls. SIMD
targets can take scalar fallbacks below their crossovers, and sparse scheduling remains Kotlin even when
arithmetic uses SIMD. Vendor rows name the entry point the binding reached. JVM vendor timings include
array transfers.

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

Pass this object as `bench`. Parameters follow the capture options; paths are relative to the repository.
The adapter owns the output directory and retains logs and partial results on failure. It also accepts
`platform`, `memory-mib` and `timeout-seconds`; Windows workers need Git Bash or an absolute KOBLAS_BASH.
Use automatic placement. Keep all arms of a comparison row on one machine in one worker attempt;
replacement attempts rerun the complete unfinished row. Other rows may use different compatible machines.
See [LabAdapter.kt](src/jvmMain/kotlin/com/eignex/koblas/bench/LabAdapter.kt) for accepted parameters and validation.

### Score lab comparisons

The scripts under [lab](lab) read fetched job directories, whatever their case layout, and need only Bash and awk:

```bash
# Ratio of jvm-simd to openblas per case, each time the median over the job's passes.
koblas-bench/lab/compare.sh results/run-1 jvm-simd openblas >run-1.txt
# Median of each case over repeated jobs.
koblas-bench/lab/median.sh run-1.txt run-2.txt run-3.txt >median.txt
# Win or lose per routine and size band.
koblas-bench/lab/score.sh openblas median.txt
```

Separate jobs land on different hosts: on cloud x86 hosts one commit's ratios differ by tens of percent per
routine between jobs, and the vendor's own times move with the host. To compare a variant, put it and its
base in one job as `configs` arms with `repeats` of at least 3, so that each row runs every arm on one
machine, and pass the arm's label as the fourth argument to `compare.sh`. It then reads only rows in which
every arm finished. `score.sh` documents its size bands and the Accelerate coprocessor exemption.
