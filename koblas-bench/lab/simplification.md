# Array BLAS boundary and simplification evidence

Baseline: main at `50c2c79dcf007080f52aa229b271b60f6c1e6395` (merged runtime-selection PR #656).
The separate backend-guide PR #657 was open when this work started. Its branch is not a dependency.
The shared primary checkout and the originating documentation worktree are excluded from this work.

## Consumers and retained surface

The Klause sources in `klause/src/commonMain/kotlin/com/eignex/klause` use the following calls.
Paths below are relative to that directory. Sizes describe source-derived fixtures and bounds, not call frequencies.

| Call site | Operation and storage | Windows and representative sizes | JVM arithmetic |
|---|---|---|---|
| `simplex/basis/DenseBasisFactors.kt`, `eliminate` | Dense GER into column-major arrays | Parent leading dimension; independent matrix/vector offsets; trailing rectangles from order at most 6 by default; controls 5, 6, 7, 34, 198 | Rank-update SIMD panels with scalar short windows |
| `lp/engine/RevisedSimplex.kt`, `restoreCachedBasicValues` | Raw dense AXPY, alpha 1 | Unit stride, offset 0, LP row count; retained corpus 2, 6, 8, 27, 56, 70 and aligned control 64 | Contiguous Vector API AXPY above one vector |
| `lp/engine/RevisedSimplex.kt`, `columnDot` | Validated CSC column dot against dense values | Nonzero CSC column offsets; column nnz; benchmark controls 8, 64, 512, 4096 | Indexed SIMD dot where hardware and count qualify |
| `simplex/basis/IndexedVector.kt`, `dot`; `simplex/basis/KotlinBasisSolver.kt`, `solveQuality` | Raw sparse indices and packed coefficient slice against dense indexed values | Separate index/value offsets and count from CSC columns | Indexed SIMD dot where hardware and count qualify |
| `simplex/basis/IndexedVector.kt`, `nrm2` | Dense values with sparse support | Index offset/count; each support index selects a dense value | Indexed SIMD square sum with finite-range rescaling |
| `lp/engine/RevisedSimplex.kt`, `sparseAxpy`; `simplex/basis/KotlinBasisSolver.kt`, `solveQuality` | Raw indexed sparse AXPY into dense workspace | Independent index/value offsets and count | Indexed SIMD updates where hardware and count qualify |
| `lp/engine/RevisedSimplex.kt`, `denseColumn` | Raw indexed sparse scatter | CSC index/value offsets/count into reused dense destination | Indexed SIMD scatter where hardware and count qualify |
| `util/SparseSlices.kt` | `SparsePrimitives` workspace scatter/gather, touched-support and ordered diagnostic helpers | Independent support/workspace offsets, markers and counts | Solver bookkeeping; preserve checked diagnostic behavior independently of BLAS arithmetic |

Klause, Combo and Kumulant source searches found no consumers of public panel/product/triangular interfaces,
engine component properties, or retained packed matrices. The benchmark module directly used those seams;
raw panel, raw product-block and retained-packing benchmark entry points are removed with that unsupported API.
Ordinary GER/GEMM and structured routines continue to exercise the internal arithmetic through public calls.

Application API: ordinary `DoubleArray`, dense/CSC containers and borrowed dense views, BLAS operations,
raw vector and indexed sparse windows, prepared sparse snapshots, and workspaces. Into operations preserve
alias safety through the backing buffer; fresh CSC results own their arrays; snapshots own immutable copies.

Internal implementation: dense panel, product and triangular contracts, cache blocks, tiles and transient
packing. Expert diagnostics and exact built-in engine selection are separate from application operations.
Explicit vendor comparisons open a binding directly. No public custom-backend SPI or retained physical layout
is supported. Portable dense and sparse Levels 1–3 remain available without a native library.

## Numerical audit

Routine-specific BLAS zero-multiplier, unused-operand, triangle and unit-diagonal rules remain requirements.
FMA, reassociated reductions and backend rounding are permitted. The scalar reference checks finite-data
accuracy, storage and ownership; it does not prescribe standard-unspecified exceptional behavior.
The [Netlib GEMM definition](https://netlib.org/lapack/explore-html/dd/d09/group__gemm_ga1e899f8453bcbfde78e91a86a2dab984.html)
states the operation and beta-zero no-read requirement without fixing multiplication order.

The baseline source already permits native multiplication order; the older benchmark guide's scaled-call
restriction is stale. Audit targets include ordered-only sparse transpose GEMV, exceptional-result comparison
tests, fixed multiplier-placement tests, and the SIMD AXPY prohibition on FMA. Norm range protection is retained:
a plain square sum can overflow or underflow for ordinary finite inputs with a representable norm. Checked
solver workspace primitives report arithmetic diagnostics deliberately and are not ordinary BLAS calls.

## Baseline and performance evidence

The baseline JVM ABI dump contains 1,178 lines. Seven route/model/reporter files contain 1,398 lines,
before embedded sparse reporting and selection diagnostics. Retained packing adds physical layout types,
three GEMM entry paths and conditional retained-panel handling to the common product schedule.

Use the accepted oneMKL evidence in [vendor-comparison.md](vendor-comparison.md), especially jobs 9039 and
9111, and the compact/streaming AXPY evidence from jobs 9068 and 9163–9164. Whole JVM Level 3 calls retain
the minimum-128 rule; whole GEMM additionally retains sides at least 16 and depth 1024–4096. Borrowed regions
do not inherit the deep-GEMM exception. Native pinning has a separate policy from JVM operand transfers.
Accelerate comparisons remain deferred. Targeted follow-up checks and results belong below as they complete.

## Historical Native policy provenance

The retained matrix work floor of 1,024 comes from the baseline source's account of a bounded local
`koblas-bench` GEMM/GEMV/TRSM ladder: exact portable Native against explicit oneMKL and OpenBLAS on one
shared desktop. That note records neither the CPU model nor a retained capture here. The explicit bindings
excluded composition validation, eligibility and operand wrapping. The floor is a conservative retained
policy, not an established crossover on other hosts.

For Level 1, the baseline `VendorVectorKernels.crossover` note attributes its cutoffs to the existing
`koblas-bench` sweep suite on pinned i9-12900H P-cores, single-threaded oneMKL 2026.1 and Kotlin/Native
2.4.10 linuxX64. It records ordinary break-evens of 40–67 elements, NRM2 at 82 and IAMAX at 100; the
retained implementation cutoffs are 64 and 128. These are historical source notes, not new timing evidence
from this simplification. The original capture is not reproduced here; CPU, library or binding changes
require their own evidence. In particular, these observations establish no Accelerate or ARM crossover.
JVM transfer policies use the separate retained lab evidence above.
