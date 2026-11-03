# Benchmark reference

Start with the [README](README.md) for commands, target names and reading results. This page covers the
settings and timing boundaries that matter when comparing results. Full options are in `capture-report.sh --help`.

## Capture settings

Use `--targets` to select implementations and `--cases` to select a workload file. `--operation` filters
that file; `--suite sweep` selects its opt-in sizes and requires an operation. Vendor targets need a
compatible installed library; the [engine guide](../BACKENDS.md) covers setup.

`--warmups` discards initial repetitions, `--samples` sets measured repetitions, and `--target-ms` sets
the time budget per repetition. `--forks` controls JVM forks. `--rounds` repeats the capture with rotating
target order; give repeated captures a new output directory. `--jvm-gc` and `--jvm-heap` fix collector and
heap; direct Gradle runs select the JDK with `-Pbench.javaVersion` or `-Pbench.expectedJavaHome`, and
`KOBLAS_BENCH_PROFILE_GC=true` retains JMH allocation measurements in target logs.

For local Linux captures, `--cpu spread` picks a lightly loaded allowed core; `--cpu auto` picks the
first allowed core. Pinning does not reserve it, and unexplained changes in controls prevent attributing
a timing difference to the code.

Windows captures run in Git Bash with JDK 25 and matching vendor DLLs on PATH, without CPU pinning.
Native Linux ARM64 captures need an executable cross-built on x64 with
`:koblas-bench:linkReleaseExecutableLinuxArm64`, passed with `--native-executable`.

## Cases

[Cases](cases.txt) use `operation+dimensions+fixture[+option=value...]`. Untagged cases belong to
`default`; `+suite=sweep` is opt-in. Compare identical case options: shape, transpose, sparse support
and reset policy all change the work being timed. Sparse matrix cases carry `mode=oneshot`, which times
the whole call.

- Ordinary mutating cases commonly include destination reset. `timing=arithmetic` excludes it where supported.
- `axpy-unit` accumulates with alpha one without a timed reset; ordinary `axpy` uses a nonunit multiplier and includes reset.
- `ger-shrinking` and `ger-window-shrinking` report a batch of `n - 1` updates, excluding matrix reset and factorization work.
- GEMM's `workspace=retained` reuses scratch; `layout=borrowed` uses offsets and padded storage.
- Sparse construction cases include allocation; view and raw indexed cases include their public validation and dispatch.
- Dense vector cases accept `stride=2`, `stride=-1` and `stride=-2`; omitted stride means contiguous owned arrays.

Kernel timings do not establish an end-to-end solver speedup. For exact boundaries, see
[Dense.kt](src/commonMain/kotlin/com/eignex/koblas/bench/Dense.kt),
[Sparse.kt](src/commonMain/kotlin/com/eignex/koblas/bench/Sparse.kt) and
[ShrinkingRankUpdate.kt](src/commonMain/kotlin/com/eignex/koblas/bench/ShrinkingRankUpdate.kt).

An engine row's `actual_kernel` names the selected engine and operation, not the body a given size reached.
A default engine that loaded a native library records its path, version and thread evidence in
`metadata.txt`. JVM vendor timings include array transfers.

## Lab captures

Use the authenticated API at lab.eignex.com through the maintained lab client. The standard manifest is
[lab/bench.json](lab/bench.json); build its JVM adapter with `./gradlew :koblas-bench:prepareLabBenchmark`.
A request identifies `repository: "Eignex/koblas"` and passes this object as `bench`:

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

Each selected workload becomes one paired case. [LabAdapter.kt](src/jvmMain/kotlin/com/eignex/koblas/bench/LabAdapter.kt)
lists the accepted parameters. Windows workers need Git Bash or an absolute `KOBLAS_BASH`.

The [calibration manifest](lab/calibration.json) prepares the grids `generateCalibrationGrids` writes under
`koblas-bench/build/calibration/`. Set `cases` to `training.txt` or `holdout.txt` for JVM, or
`native-training.txt` or `native-holdout.txt` for Native; `native-timing.txt` is a small subset for comparing
timing protocols. Use [the macOS manifest](lab/calibration-macos.json) for Accelerate and
[the Linux ARM64 manifest](lab/calibration-linux-arm64.json), which cross-builds on x64, for ARM workers.
The same manifests prepare vendor comparisons. Native executables are built once during preparation, and
measured rows never invoke Gradle.

### Recalculate routing profiles

Capture on the intended host; this example prepares Linux x64 Native comparisons:

```bash
./gradlew :koblas-bench:prepareLabCalibration :koblas-bench:tuneEngine \
  --args="--capture --runtime native --backend onemkl --backend openblas --cpu * --native-executable koblas-bench/build/bin/linuxX64/releaseExecutable/koblas-bench.kexe --output /tmp/native-profile"
```

The tool emits `Profile.kt` and external evidence. Use its declaration in the standard target configuration,
rebuild, and capture disjoint holdouts through `native-default` or `jvm-default`. Use `--runtime jvm` for
transfer-inclusive tuning, or repeated `--results PATH` arguments to refit fetched captures.
`--fallback aocl=openblas` and `--fallback armpl=openblas` label unmeasured vendor estimates.

Validate the rebuilt composition against its generated profile identity:

```bash
./gradlew :koblas-bench:tuneEngine \
  --args="--validate-results /tmp/holdouts --training /tmp/native-profile --runtime native --output /tmp/validated-profile"
```

Validation checks profile identity, disjoint cases, provider reach and paired ratios. Keep reports and
measurement data outside the repository.

### Score lab comparisons

The scripts under [lab](lab) read fetched job directories and need only Bash and awk:

```bash
# Ratio of jvm-simd to openblas per case, paired within each pass and weighted equally over CPU models.
koblas-bench/lab/compare.sh results/run-1 jvm-simd openblas >run-1.txt
# Several jobs: pool their cases in one directory so the weighting spans all of them.
mkdir -p pooled && cp -r results/run-1/cases pooled/run-1 && cp -r results/run-2/cases pooled/run-2
koblas-bench/lab/compare.sh pooled jvm-simd openblas >pooled.txt
# Win or lose per routine and size band.
koblas-bench/lab/score.sh openblas pooled.txt
```

One lab pool can mix CPU models whose ratios differ, so `compare.sh` weighs each model equally. `median.sh`
combines comparisons without that weighting; use it only for runs on one CPU model.

To compare a variant, put it and its base in one job as `configs` arms with `repeats` of at least 3 and pass
the arm's label as the fourth argument to `compare.sh`, which then reads only rows in which every arm finished.
`score.sh` documents its size bands and the Accelerate coprocessor exemption.
