# Kernel transition semantic baseline

This inventory records current contracts at production base `12628d18`, before the SME port. Oracle ownership
is explicit so later optimized kernels can be checked independently. `ReferenceBlas` in
`koblas/src/commonTest/kotlin/com/eignex/koblas/dense/DenseVectorKernelsConformance.kt` is
`BuiltinEngines.scalar`, composed from scalar vector/panel and portable packed kernels. Test helpers use this
engine explicitly; the default `koblas` engine is not an independent oracle.

All paths below are under `koblas/src/`; test owners are in `commonTest/kotlin/com/eignex/koblas/` unless stated.
Existing tests own most contracts. PR 01 adds strided poisoned-output/guard and pre-mutation overlap-rejection
coverage plus full-buffer benchmark block conformance across old geometries and timing modes.

| Contract | Portable semantic owner | Test/oracle owner |
| --- | --- | --- |
| Alpha zero suppresses operand products; beta zero overwrites without reading old destination | `dense/DenseBlas.kt`, `DenseGemvKernels.kt`, `BuiltinBlas.kt`, `DenseOrderedProductKernels.kt` | `dense/BlasTest.kt`, `SymmetricBlasTest.kt`, `StridedBlasTest.kt`; explicit scalar engine and poisoned NaN/Inf operands |
| Empty GEMV is a quick return; zero-depth GEMM preserves shape and applies its scaling contract | `DenseBlas.kt`, dense GEMV/GEMM access loops | `dense/BlasTest.kt`, `StridedBlasTest.kt`; zero-depth/empty tests and backing-storage assertions |
| Standalone AXPY alpha zero is a no-read shortcut; matrix panel arithmetic evaluates zero products where specified | `ScalarVectorKernels`, `ScalarPanelKernels`, ordered dense loops | `dense/DenseVectorKernelsConformance.kt`, `BlasTest.kt`, `SymmetricBlasTest.kt`; zero times infinity and NaN classification |
| Robust nrm2 avoids intermediate square overflow/underflow and propagates non-finites | scalar vector norm implementation and strided norm traversal | `dense/DenseVectorKernelsConformance.kt`, `StridedBlasTest.kt`, `MatrixNormsTest.kt`; explicit scale/ssq oracle and classification checks |
| Selected triangles are the only input read/output written; unstored triangles may be poisoned | symmetric/triangular dense kernels and packed panel structures | `dense/SymmetricBlasTest.kt`, `GemmtTest.kt`, `TriangularTest.kt`, `PackedPanelsTest.kt`; expanded full matrices against scalar GEMM/GEMV and untouched triangle checks |
| Unit diagonals are materialized/used as one without reading stored values | `PackedLayout.kt`, `PackedTriangularSolve.kt`, triangular scalar loops | `dense/PackedPanelsTest.kt`, `PackedTrsmTest.kt`, `TriangularTest.kt`; poisoned diagonal and selected-triangle fixtures |
| Triangular division, zero coefficients and source pivots preserve ordered semantics | portable TRSV/TRSM/TRMM and packed solve loops | `dense/TriangularTest.kt`, `PackedTrsmTest.kt`, `PackedTriangularSolveTest.kt`; singular/subnormal division, transpose-specific zero RHS behavior, underflow then subsequent products |
| Reassociation and scaling placement may overflow despite finite inputs; decline packed fast paths before mutation | finite-product eligibility and ordered dense symmetric/triangular fallbacks | `dense/SymmetricBlasTest.kt`, `TriangularTest.kt`; finite scaling overflow, rank-two interleaving, exceptional products and workspace-return tests |
| Permitted owning/Into aliases stage source storage before mutation; packed panel alias staging uses workspace | public operations, packed panels and structured update entry points | `MatrixOpsTest.kt`, `VectorOpsTest.kt`, `dense/PackedPanelsTest.kt`, `SymmetricBlasTest.kt`; compare actual backing arrays, not detached copies |
| Borrowed GEMV/GEMM views reject overlapping destinations before mutation; disjoint views can share buffers | `dense/DenseBlas.kt`, `StridedViews.kt` overlap validation | `StridedViewOverlapTest.kt`, `dense/StridedBlasTest.kt`; new reversed-vector rejection checks every backing bit even with beta zero |
| Offsets, leading dimensions, negative vector strides and untouched surrounding storage | strided access loops and kernel window contracts | `StridedViewsTest.kt`, `dense/StridedBlasTest.kt`, `DenseVectorKernelsConformance.kt`; new NaN-filled strided C beta-zero case preserves signed-zero guards |
| Full physical tiles, logical edges, selected output masks, depth tails and positive-zero padding | `dense/PackedKernels.kt`, `PackedTileOutput.kt`, `PackedLayout.kt` | `dense/PackedGemmTest.kt` written-out product; `PackedTrsmTest.kt` reference composition; `PackedPanelsTest.kt` round trips and padding bit checks |
| Packed raw kernels do not permit operand/output aliasing; public panel helpers stage permitted overlap | `dense/PackedKernels.kt` and `PackedPanels.kt` KDoc | `dense/PackedPanelsTest.kt`; do not generalize public helper alias support to raw kernels |
| Sparse canonical CSC, implicit zeros, structural selection and retained workspace semantics | sparse reference algorithms and indexed scalar kernels | `sparse/ReferenceSparseBlas.kt`, `SparseBlasTest.kt`, `SparseTriangularTest.kt`, `PreparedSparseMatrixTest.kt`, `SparseSlicesTest.kt` |

Finite conformance uses existing relative/absolute tolerances, not bitwise equality of differently associated
sums. NaN/infinity require classification assertions; signed zero, padding and untouched guards require bit
checks when contractual. `wellConditioned` has identity pivots; permutation tests must perturb it first.
No fast-math, approximate reciprocal, production selection or numerical semantics change belongs to this PR.

The benchmark's `gemm-add-v1` fixture is finite and therefore does not replace the exceptional-value suites.
Its `PackedConfigurationTest` compares every output element with the explicit scalar full GEMM, repeats calls
to catch accumulation/reset errors, and tests both retained and repacked inputs. Incompatible tile geometries
return unsupported; they do not create an arbitrary-width scalar substitute.
