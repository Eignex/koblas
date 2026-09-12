# koblas-bench

CPU benchmarks for Koblas, OpenBLAS, Accelerate, and oneMKL. Requires JDK 25 and a C compiler.

```bash
# Full capture: JVM scalar, C, SIMD, native, and available platform vendors.
koblas-bench/capture-report.sh --samples 10 --warmups 5 --target-ms 200 --forks 2

# First three selected cases, one short sample, no warmup.
koblas-bench/capture-report.sh --smoke

# Vendor-only run.
koblas-bench/capture-report.sh --vendors-only --libraries openblas,accelerate
```

Full reports go to `reports/<hardware-sha256>/<run-id>/`; smoke and vendor-only runs default to `build/benchmarks/`.
Use `--output NEW_DIR` to override. Each report contains one CSV per target and `metadata.txt` with hardware,
source revision, timing settings, execution timestamps, and actual runtime/library configurations.
Each CSV records one row per case with sample/fork counts, median, minimum, and maximum ns/op.
CSV files contain only case definitions, status, actual kernel and measured results; run settings and runtime
identity belong exclusively to `metadata.txt`. Capture is the only reporting script.

Use `--native-variant scalar|sse2|avx2|neon` for exact raw C arithmetic in the JVM C and Native targets.
Unavailable variants fail the capture. Kotlin scalar and JVM SIMD targets retain their distinct identities.

[`cases.txt`](cases.txt) defines the workload. Filter with `--operation NAME` or `--suite packed`.
Compare matching cases and timing boundaries; prepared, one-shot, and packing-inclusive timings differ.
Unsupported cases have no timing. A selected target failure stops capture; existing reports are never overwritten.

For `scal` and `spgather`, `+timing=arithmetic` excludes resets; arithmetic scaling uses alpha = -1.
Default gather resets only the dense source; older captures also reset the sparse output, so compare those separately.

Vendors use one thread. `--libraries all` selects OpenBLAS and Accelerate on macOS, OpenBLAS and oneMKL on Linux.
Homebrew OpenBLAS is detected automatically. Accelerate requires macOS 15+ and includes sparse vectors and
matrix products. For Linux oneMKL, set `ONEMKL_LIBRARY` to its runtime library path.
