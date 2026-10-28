# JVM kernels and oneMKL

Public storage remains ordinary `DoubleArray`. These comparisons include the JVM binding's transient
native staging and writeback. No native-backed containers or critical vendor downcalls are introduced.

The analysis is limited to oneMKL on the measured Intel host. The Accelerate comparison was skipped at
the user's request; these results establish no Apple performance crossover.

## Implementation decisions

| Area | Selected policy |
|---|---|
| Large GEMM, GEMMT, SYMM, SYRK, SYR2K, TRMM and TRSM | Prefer eligible oneMKL whole calls; keep owned fallback and exact engines |
| DOT, SUM, ASUM and NRM2 | Keep explicit JVM SIMD and its numerical fallbacks |
| Simplex GER, indexed dot/norm, sparse AXPY/scatter and dense AXPY | Keep JVM SIMD paths; native composition preserves their owned kernel arms |
| IAMAX | Keep owned SIMD; further tuning is lower priority than simplex work |
| SCAL, SWAP, ROT and COPY | Keep ordinary loops and array-copy intrinsics; no dedicated Vector API replacement |
| Eight-column product prototype | Removed because it introduced per-call allocation |
| Deep thin GEMM | Prefer oneMKL for whole matrices with both output extents at least 16 and depth 1024 through 4096; borrowed views keep the larger cutoff |

The availability floor remains portable dense and sparse BLAS. Selecting a profitable native call does
not remove the owned implementation needed on other hosts or by an explicitly selected engine.

## Evidence

| Lab job | Source | Purpose | Status |
|---|---|---|---|
| [9002](https://lab.eignex.com/jobs/9002) | `e1dc62f0` | 73-case scalar/SIMD/JVM oneMKL/native oneMKL diagnostic | Complete, three rotated rounds |
| [9012](https://lab.eignex.com/jobs/9012) | `ee7b7f10` | Longer warmup and default-policy comparison | Failed reference check on the old 768-row triangular fixture; partial timings retained |
| [9039](https://lab.eignex.com/jobs/9039) | `a1d6ada5` | Conditioned fixtures, unrolled AXPY and actual default routing | Complete; use rounds 1 and 2 for timing |
| [9062](https://lab.eignex.com/jobs/9062) | `fe0deefe` | Wider-tile follow-up | Cancelled because concurrent captures pin the same CPU |
| [9063](https://lab.eignex.com/jobs/9063) | `a16e61fd` | Norm follow-up | Cancelled; follow-ups will run sequentially |
| [9066](https://lab.eignex.com/jobs/9066) | `a23a3736` | Sequential combined tile, allocation and norm follow-up | Complete; wider tile rejected |
| [9114](https://lab.eignex.com/jobs/9114) | `b34eaa8a` | Supplementary timing of the broader candidate cutoff | Cancelled before measurement after narrowing the cutoff |
| [9111](https://lab.eignex.com/jobs/9111) | `d2a9a219` | Thin GEMM neighboring shapes, transposes and retained workspaces | Complete, three rounds |
| [9068](https://lab.eignex.com/jobs/9068) | `57fc0f72` | Compact and streaming AXPY through exact and default engines, with allocation profiling | Complete, three rounds |
| [9070](https://lab.eignex.com/jobs/9070) | `59bcc0e8` | 82-case comparison of the final implementation against Accelerate on Apple hardware | Cancelled at user request before measurement |

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

Jobs 9006, 9040 and 9064 never started measurement and were replaced by the corrected 82-case Apple
workload at the final runtime revision.
The final Apple run, 9070, was then cancelled at the user's request before measurement.
The final workload admits value-dependent dense NRM2 as a composition, and raw sparse slices use the
explicit engine rather than requiring identity with the default. Vendor arms honestly report no symbol
for SUM or indexed sparse norms.

## oneMKL dispatch

The exact scalar and SIMD engines stay independent of installed libraries. The JVM default retains their
owned kernels and can compose oneMKL on Intel Linux when native access is explicitly enabled. Only the
seven Level 3 routines above qualify when every output extent and depth reaches 128. Whole-matrix GEMM
also qualifies with both output extents at least 16 and depth from 1024 through 4096. Borrowed views
keep the larger cutoff because their additional packing was not included in the thin-product timings.
Other thin products, few-right-hand-side solves, Level 1/2 calls, windowed GER and retained packed products
keep owned kernels.
Zero-multiplier rules and alias staging remain with the shared host composition.

The 128 boundary is supported by the two accepted timing rounds of job 9039. Accelerate is outside this
analysis. Apple defaults remain unchanged; Intel measurements do not determine their routing.

## Retained oneMKL dispatch comparison

Job 9039 passed its numerical checks and completed all three rounds. Rounds 1 and 2 finished at
16:40:07 UTC before follow-up measurements began. Round 3 overlapped follow-up jobs pinned to the
same CPU and is excluded from performance conclusions. The table uses medians of the two accepted
round medians, with the same ordinary-array call and transfer costs in every JVM arm.

| Routine | Default speedup over exact SIMD at order 128 | At 256 | At 768 |
|---|---:|---:|---:|
| GEMM | 2.17x | 2.15x | 1.92x |
| GEMMT | 1.81x | 1.95x | 1.45x |
| SYMM | 2.18x | 2.60x | 1.77x |
| SYRK | 2.19x | 1.78x | 1.91x |
| SYR2K | 3.17x | 2.09x | 1.59x |
| TRMM | 4.34x | 3.08x | 2.89x |
| TRSM | 4.66x | 3.10x | 2.36x |

AXPY at 65,536 elements measured about 38 microseconds for exact SIMD and 37 microseconds for native
oneMKL, while the JVM oneMKL arm took 143 microseconds. At 4,096 elements the four-block SIMD body
lost to the scalar loop (2.20 versus 1.88 microseconds); the subsequent implementation restores the
compact SIMD body below 65,536 and retains unrolling for streaming calls. The separate follow-up
validates that split. Default AXPY showed substantial fork variation, so these figures do not establish
a precise performance guarantee for its surrounding compilation.

ASUM and IAMAX at 4,096 measured 0.29 and 0.36 microseconds in exact SIMD. Their JVM oneMKL arms took
1.80 and 1.72 microseconds. IAMAX at 65,536 still trails native compute (8.7 versus 4.0 microseconds),
but the JVM binding takes about 29.6 microseconds. Dropping the owned body for that binding would regress
the ordinary-array call.

The initial deep thin GEMM controls favored the JVM vendor arm. Job 9111 then checked neighboring
shapes and retained workspaces before expanding the whole-matrix GEMM policy. Single-output products
remain owned; their transposed-left path can already use a SIMD reduction, and no shape-only native
crossover has been established across those orientations.

## Thin GEMM crossover

Job 9111 compared exact SIMD with JVM oneMKL, including transfers, on the same Xeon Platinum 8375C.
It used three rotated rounds, eight 200 ms warmups, five 200 ms samples and allocation profiling.
The retained workspace is supplied only to the owned call because the vendor binding has no such
parameter. Both arms use the same arrays, flags, arithmetic and destination reset.

| Shape and flags, with retained workspace | Owned SIMD | JVM oneMKL | Speedup |
|---|---:|---:|---:|
| 16x16x1024 | 49.90 us | 23.45 us | 2.13x |
| 16x16x4096 | 208.38 us | 118.85 us | 1.75x |
| 16x16x4096, both operands transposed | 225.06 us | 159.67 us | 1.41x |
| 16x64x4096 | 626.16 us | 450.46 us | 1.39x |
| 64x16x4096 | 690.03 us | 464.81 us | 1.48x |
| 32x32x4096 | 523.62 us | 405.37 us | 1.29x |
| 16x16x16384 | 854.54 us | 750.04 us | 1.14x |

Workspace reuse did not erase the native advantage. The 16x16x16384 case won in every round, but its
smaller margin does not justify extending the new rule beyond depth 4096, where transpose controls
also establish a native advantage. Outputs narrower than 16, depths outside 1024 through 4096, and borrowed views keep their previous routes unless they satisfy
the existing large Level 3 rule. Tests cover both cutoffs, each adapted operand, route attribution and
the destination's complete backing buffer.

A local numerical smoke capture against the checksum-pinned oneMKL 2026.1 package verified the final
default at depths 1024 and 4096, a both-transposed case at 4096, and excluded side/depth controls.
All six cases passed preflight;
the first three reported `onemkl/cblas_dgemm` and all three controls reported owned products. Its short
warmup is not performance evidence. All eight default-engine tests also passed with the real library
loaded, including alias handling and preservation of the owned dense/indexed kernel arms. The optional
supplementary lab timing run 9114 was cancelled before measurement because it targeted the broader
cutoff; it is not used for these conclusions.

## Final AXPY validation

Job 9068 validates the compact/streaming split in three rotated rounds with GC profiling. These paired
figures supersede the earlier AXPY diagnostics for the final implementation:

| Elements | Scalar JVM | Exact SIMD JVM | Default JVM | JVM oneMKL | Native oneMKL |
|---|---:|---:|---:|---:|---:|
| 64 | 0.023 us | 0.030 us | 0.025 us | 0.191 us | 0.081 us |
| 4,096 | 1.745 us | 2.128 us | 1.745 us | 5.704 us | 1.722 us |
| 65,536 | 40.975 us | 44.016 us | 44.618 us | 146.969 us | 35.312 us |
| 65,537 | 40.225 us | 43.946 us | 44.023 us | 147.744 us | 36.613 us |

The default cache-resident case is close to native, while the streaming case retains a roughly 25%
compute gap and exact SIMD can lose modestly to HotSpot's scalar-loop compilation. All retain the
required explicit JVM SIMD path. The native JVM binding still loses decisively after transfers.
The earlier 38 us exact-SIMD result is not a repeatable promise for the surrounding compilation.

GC profiling measured about 1.3 bytes per streaming call for SIMD/default versus about 1.2 for scalar,
consistent with harness sampling overhead rather than per-vector allocation. The separate uninstrumented
allocation probes measured zero bytes for compact and streaming bodies, including offsets and tails,
at the preferred/narrow widths and with FMA disabled. The default's 65,536-element round medians were
45.8, 44.6 and 44.3 us; the earlier 93 us fork was not reproduced.

## Wider-tile and norm follow-up

Job 9066 completed three sequential rotated rounds with allocation profiling. The eight-column FMA
prototype improved GEMM at 128, 256 and 768 by roughly 17%, 26% and 18%, but lost about 4% at order 64.
A raw `64x64x128` packed-block call allocated about 36,878 bytes in the prototype versus about 14 bytes
of profiler noise in the four-column body. GEMM 128 allocated about 410 kB versus 262 kB; at 256,
1.57 MB versus 0.39 MB. The prototype is removed rather than promoted. Its original revision retains
all code needed to reproduce the experiment; the current profiler remains available independently.

The norm controls also favor owned arithmetic after transfer costs. At length 4,096, exact SIMD NRM2
took 0.56 microseconds versus 1.25 for native oneMKL and 2.55 for its JVM binding. SUM took 0.28 versus
4.66 microseconds scalar. At 65,536, indexed norm took about 5.0 versus 6.8 microseconds scalar.
SUM and indexed norms have no corresponding vendor symbol in these bindings and are reported as
unsupported on those arms. These figures use medians of the three round medians; norm attribution
retains the scalar rescaling possibility rather than calling it an exact SIMD-only measurement.

## JVM coverage

| Family | Explicit Vector API arithmetic | Owned scalar cases |
|---|---|---|
| DOT, SUM, ASUM, AXPY | Contiguous runs of at least one lane block | Short and strided runs |
| NRM2 | Contiguous square-sum fast path | Stable rescaling when magnitudes require it; short and strided runs |
| IAMAX | Contiguous runs of at least 256 elements and one lane block | Short and strided runs |
| COPY | Contiguous public copies use the JVM array-copy intrinsic | Strided or sparse copies use ordinary loops |
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
public storage model. Ordinary JNI `GetDoubleArrayElements` may return either a pinned array or a copy;
it provides no portable zero-copy guarantee. See the [JNI array access contract](https://docs.oracle.com/en/java/javase/25/docs/specs/jni/functions.html#getprimitivetypearrayelements-routines)
and the [JDK 25 critical-call contract](https://docs.oracle.com/en/java/javase/25/docs/api/java.base/java/lang/foreign/Linker.Option.html#critical(boolean)).

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
