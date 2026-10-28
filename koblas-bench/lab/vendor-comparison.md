# JVM kernels and installed BLAS

Public storage remains ordinary `DoubleArray`. These comparisons include the JVM binding's transient
native staging and writeback. No native-backed containers or critical vendor downcalls are introduced.

## Evidence

| Lab job | Source | Purpose | Status |
|---|---|---|---|
| [9002](https://lab.eignex.com/jobs/9002) | `e1dc62f0` | 73-case scalar/SIMD/JVM oneMKL/native oneMKL diagnostic | Complete, three rotated rounds |
| [9012](https://lab.eignex.com/jobs/9012) | `ee7b7f10` | Longer warmup and default-policy comparison | Failed reference check on the old 768-row triangular fixture; partial timings retained |
| [9039](https://lab.eignex.com/jobs/9039) | `a1d6ada5` | Conditioned fixtures, unrolled AXPY and actual default routing | Pending |
| [9040](https://lab.eignex.com/jobs/9040) | `a1d6ada5` | Full comparison against Accelerate on Apple hardware | Pending Mac capacity |

Job 9002 used an Intel Xeon Platinum 8375C, eight double lanes, JDK 25.0.4.1, Kotlin 2.4.10, Serial GC and
a fixed 512 MiB JVM heap. It used one JMH fork per round, three 100 ms warmups and three 100 ms samples.
Reduction logs showed continuing compilation during measurement, so their apparent regressions do not
establish steady-state performance. The retained comparisons use eight 200 ms warmups and five 200 ms
samples per fork, with three rotated rounds. All BLAS invocations use one compute thread.

The diagnostic's Level 3 medians suggest a real whole-call routing opportunity even after staging:

| Routine | JVM oneMKL speedup over owned kernels at order 256 | At order 768 |
|---|---:|---:|
| GEMM | 1.85x | 1.52x |
| GEMMT | 1.60x | 1.75x |
| SYMM | 2.30x | 2.47x |
| SYRK | 1.80x | 1.92x |
| SYR2K | 1.87x | 1.71x |
| TRSM | 3.07x | 2.45x |
| TRMM | 3.16x | 2.82x |

These are medians of the three round medians, for the matched cases in
`cases-vendor-comparison.txt`. They are diagnostic evidence, not the final default-policy validation.
The initial DOT and GEMV comparisons were already near the native oneMKL arm; their JVM vendor arms
lost after operand transfer. A vendor's arithmetic time alone therefore cannot select a JVM route.

Dense triangular fixtures now keep each off-diagonal row norm below one half. This bounds conditioning
with explicit or implicit unit diagonals without relaxing scalar-reference tolerances. Cross-revision
triangular timings with the old fixture are not paired comparisons.

## Dispatch candidate

The exact scalar and SIMD engines stay independent of installed libraries. The JVM default retains their
owned kernels and can compose oneMKL on Intel Linux when native access is explicitly enabled. Only the
seven Level 3 routines above qualify, and every output extent and depth must reach 128. Thin products,
few-right-hand-side solves, Level 1/2 calls, windowed GER and retained packed products keep owned kernels.
Zero-multiplier rules and alias staging remain with the shared host composition.

The 128 boundary is a conservative candidate under validation by job 9039. Accelerate default dispatch
awaits Apple measurements. The code does not infer Apple performance from Intel measurements.

## JVM coverage

| Family | Explicit Vector API arithmetic | Owned scalar cases |
|---|---|---|
| DOT, SUM, ASUM, AXPY | Contiguous runs of at least one lane block | Short and strided runs |
| NRM2 | Contiguous square-sum fast path | Stable rescaling when magnitudes require it; short and strided runs |
| IAMAX | Contiguous runs of at least 256 elements and one lane block | Short and strided runs |
| SCAL, SWAP, ROT | No explicit Vector API body; ordinary element loops permit HotSpot vectorization | Same loops on every JVM |
| GEMV, SYMV, GER, SYR, SYR2, TRMV, TRSV | Shared dot, update and coupled panels, including transpose variants | Panel edges, narrow and strided windows; sequential substitution dependencies |
| GEMM, GEMMT, SYMM, SYRK, SYR2K | Packed product tiles or unpacked panels | Tile edges and small windows |
| TRMM, TRSM | Product windows and diagonal operations across right-hand sides | Narrow right-hand-side groups and edges |
| Indexed sparse dot, norm, gather, scatter, AXPY | Indexed bodies where the configured hardware and width gates qualify | Sparse-sparse dot and unsupported indexed hardware; scheduling stays portable |

The library does not implement every BLAS storage family: banded and packed-triangle BLAS are not part
of this inventory. A scalar leaf in a shared schedule is not a missing whole-call implementation.

## Ordinary arrays and native transfers

The owned JVM paths read and write the caller's arrays directly. Product panel packing and snapshots
needed for overlapping operands are separate algorithm costs. Host bindings currently stage array
operands in native memory and copy destinations back; the timed JVM vendor arm includes both costs.

The JDK's heap-segment escape hatch, `Linker.Option.critical(true)`, requires a target that is extremely
short in every case. Arbitrary vendor BLAS calls do not meet that blanket contract. Consequently this
work does not remove transfer costs by pinning arbitrary arrays across vendor execution or change the
public storage model. See the [JDK 25 critical-call contract](https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/lang/foreign/Linker.Option.html#critical(boolean)).

## Klause simplex coverage

| Consumer operation | JVM SIMD path retained |
|---|---|
| Windowed GER in dense basis elimination | SIMD rank-update panels; shrinking scalar tails are reported |
| Dense AXPY in revised simplex | Explicit contiguous Vector API body, including unrolled updates |
| Indexed sparse dot and selected CSC column dot | Indexed SIMD reductions where hardware and support width qualify |
| Indexed norm | SIMD square-sum with the required scalar rescaling fallback |
| Sparse scatter and AXPY | Indexed SIMD bodies with existing hardware and width gates |

Short or strided calls can still select scalar bodies. A function having a SIMD path does not make every
call of it vectorized. Sparse workspace bookkeeping and pivot-selection primitives have no BLAS vendor
entry point and remain portable.

## Reproduce

Submit `koblas-bench/lab/vendor-comparison.json` through the Lab Bench adapter with `vendor=onemkl` or
`vendor=accelerate`. The adapter freezes source and workload inputs, selects the execution platform and
links its native reference before timing. oneMKL is extracted privately from the checksum-pinned 2026.1
reference package; Accelerate is the host's system framework.

`cases` selects a file under `koblas-bench/lab/`. Other controls are `operation`, `targets`, `rounds`,
`warmups`, `samples`, `target-ms` and `forks`. Default targets are scalar JVM, SIMD JVM, JVM vendor and native
vendor. Include `jvm-default` when validating policy. Linux captures pin one allowed CPU; pinning does not
reserve that CPU. Reports retain the exact routes, library identity, runtime, source and binary hashes,
sample spread and every completed round.
