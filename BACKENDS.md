# Backends and native libraries

Koblas works without an installed BLAS library. Applications use ordinary `DoubleArray` storage and the
default engine; backend selection is automatic.

## What should I install?

| Platform and workload | Setup |
|---|---|
| Desktop JVM, vectors, sparse BLAS or small matrices | JDK 25+ with `--add-modules=jdk.incubator.vector` enables owned SIMD. No native library is needed. |
| Desktop JVM, large dense matrix operations on Intel Linux | On JDK 25+, optional oneMKL can help. Make its runtime discoverable. For classpath applications add `--enable-native-access=ALL-UNNAMED`; for module-path applications use `--enable-native-access=com.eignex.koblas`. |
| Other desktop JVM hosts | The default uses owned kernels. Installing AOCL, ArmPL, OpenBLAS or Accelerate does not currently enable automatic JVM native dispatch. |
| Kotlin/Native on Linux | Optional oneMKL on Intel, AOCL on AMD or ArmPL on ARM64, with OpenBLAS as fallback. |
| Kotlin/Native on macOS | Accelerate is supplied by the OS. |
| JDK 17–24, Android and iOS | Portable Kotlin; no native library is needed. |

Every supported operation has a portable fallback. The Vector API and native-access flags are independent;
a JVM host call does not require the Vector API module. Native library lookup and engine selection resolve
once, so configure the process before first use. Install a supported LP64 CBLAS runtime through its
vendor or distribution, and expose its shared library on the loader search path. Standard oneAPI
`~/intel/oneapi/mkl/latest` and `/opt/intel/oneapi/mkl/latest` prefixes are also searched.

JVM bindings stage ordinary arrays into native memory and copy destinations back; arbitrary `DoubleArray`
arguments have no zero-copy foreign-call guarantee. Owned kernels read the arrays directly.

## What is supported?

| Functions | Default JVM implementation | Optional automatic JVM native acceleration |
|---|---|---|
| DOT, SUM, NRM2, ASUM, IAMAX, AXPY | Owned SIMD, with scalar fallbacks for short/strided calls and finite-range norm rescaling. | No. |
| SCAL, COPY, SWAP, ROT | Owned loops and array-copy intrinsics; HotSpot can optimize the loops. | No. |
| GEMV, SYMV, GER, SYR, SYR2, TRMV, TRSV | Owned matrix scheduling with SIMD panels where applicable. | No. |
| GEMM, GEMMT, SYMM, SYRK, SYR2K, TRMM, TRSM | Owned SIMD with portable edges and fallbacks. | oneMKL on eligible Linux x64 hosts, under the rules below. |
| Sparse vector and CSC matrix BLAS | Owned structure handling with SIMD numerical kernels where applicable. | No whole sparse call is handed to a vendor. |

Transpose, triangle and side variants are included where the API defines them. Windowed GER stays on the owned schedule. Product packing is transient and internal. Banded/packed-triangle BLAS interfaces and
LAPACK factorizations are outside the current API; installing a library does not add them.

## Current selection thresholds

On the JVM, automatic native selection currently applies only when oneMKL is the first platform preference:
Intel or unclassified Linux x64 CPUs. AMD prefers AOCL, so its JVM default stays owned. Other vendors remain
available through explicit bindings on supported hosts.

| Native JVM call | Eligible shape |
|---|---|
| The seven dense Level 3 operations above | Logical rows, columns and shared depth are all at least 128. |
| Additional whole-matrix GEMM cases | Both output sides are at least 16 and depth is 1,024–4,096. Borrowed views keep the larger rule. |

These are current implementation choices, not requirements applications must tune. Missing libraries or
symbols retain the owned route. Kotlin/Native has separate foreign-call thresholds; they do not apply to
JVM array transfers. Kernel lane counts, packing sizes and unrolling choices are backend internals.

## How much does native help?

The retained Intel comparison measured the default engine using oneMKL against the exact owned SIMD
engine at matrix orders 128, 256 and 768. These are the minimum and maximum speedups across those three
tested orders, including JVM transfers into native memory and destination writeback:

| Routine | Observed oneMKL speedup over owned JVM SIMD |
|---|---:|
| GEMM | 1.92–2.17x |
| GEMMT | 1.45–1.95x |
| SYMM | 1.77–2.60x |
| SYRK | 1.78–2.19x |
| SYR2K | 1.59–3.17x |
| TRMM | 2.89–4.34x |
| TRSM | 2.36–4.66x |

The [accepted comparison](koblas-bench/lab/vendor-comparison.md#retained-onemkl-dispatch-comparison)
records the exact cases and accepted rounds. A separate
[deep-GEMM comparison](koblas-bench/lab/vendor-comparison.md#thin-gemm-crossover)
measured 1.29–2.13x for selected whole matrices with retained workspace. These measurements support the
current selection rules; they are not guaranteed gains for other shapes, hosts or vendors.

Owned JVM kernels read ordinary arrays directly. The measured JVM vendor transfers made several vector
calls slower, which is why installing a library does not redirect those calls. General Level 2, sparse and
other-platform performance gaps have not been established systematically. Supported does not mean
benchmarked across every shape and host.

The [measurement report](koblas-bench/lab/vendor-comparison.md) contains exact cases, hardware, ratios,
threshold sources and limitations. It is evidence for the implementation, rather than application setup.
Engine route queries and exact engines are directly callable. They report scheduling, actual
leaves, whole-call native execution and relevant copies; a selected SIMD engine does not mean every sparse
operation vectorizes. Normal application code uses the default engine without inspecting these internals.

## Numerical and storage contracts

Arithmetic follows each routine's ordinary BLAS semantics. Implementations may use FMA, tree reductions,
blocking and different rounding. There is no reproducibility or uniform NaN/Inf, signed-zero or tied-index
promise beyond the routine's requirements. Norms retain finite-range scaling. Zero-multiplier no-read,
selected-triangle and unit-diagonal rules remain part of their routines.

Storage guarantees are separate: Into calls support backing-buffer aliases, CSC stored positions participate
while absent positions are never evaluated, fresh sparse results own their arrays, and prepared snapshots own
immutable copies. Concurrent readers supply separate destinations and workspaces. Raw vector and indexed
sparse offset/count APIs remain available without allocating container wrappers.
