# koblas

Dense and sparse linear algebra for Kotlin Multiplatform.

Kotlin has no standard multiplatform BLAS/linear-algebra library. koblas is a small, focused one:
serializable matrix and vector containers, arithmetic over them, dense and sparse factorizations, and a
runtime-swappable compute backend a tuned BLAS/LAPACK (and, later, GPU) implementation can drop into
without touching callers.

## What it provides

Containers — a read-only `View` contract; concrete backings also expose their storage for in-place work:

- `MatrixView` / `DenseMatrix` — flat, row-major `DoubleArray` backing.
- `VectorView` / `DenseVector` / `SparseVector` — dense or compressed-sparse.
- `SparseMatrix` — compressed-sparse-column (CSC) with matrix–vector products.

Dense arithmetic — free functions over the views (BLAS-1/2) plus the swappable backend (BLAS-2/3 +
factorizations):

- BLAS-1/2 free functions: `dot`, `axpy`, `scale`, `norm2`, `asum`, `iamax`, `copy`, `swap`,
  `addOuter` (rank-1 update), `matVec`, `transpose`, `forEachStored`.
- Triangular solves: `trsv` / `trsm` over either triangle, with transpose and unit-diagonal flags.
- `LinearAlgebra` backend: `gemv` / `gemm` (full BLAS alpha/beta/transpose forms plus fresh-result
  conveniences), `syrk`, and general LU `factor` / `solve` (with a transpose flag for BTRAN) with a
  `determinant`.
- Ergonomic entry points: `DenseMatrix.lu()`, `LuDecomposition.solve(b, transpose)`, `DenseMatrix.matMul(other)`.
- Symmetric positive-definite: `cholesky`, `choleskyUpdateInPlace` / `choleskyDowndateInPlace`,
  `solveSpd`, `invertSpd`.

Sparse linear algebra:

- `SparseLu` — Markowitz-pivoting sparse LU with `O(nnz)` FTRAN/BTRAN solves and a determinant.
- `EtaBasis` — product-form-of-the-inverse basis for `O(m)` rank-1 updates between refactorizations
  (the kernel a sparse simplex builds on).

All view containers are `@Serializable` (kotlinx.serialization); `DenseMatrix` serialises to a 2D
`Array<DoubleArray>` wire form independent of its flat backing.

## Usage

```kotlin
import com.eignex.koblas.*

// Dense solve A·x = b via LU (general, non-symmetric).
val a = DenseMatrix.of(
    arrayOf(
        doubleArrayOf(2.0, 1.0),
        doubleArrayOf(1.0, 3.0),
    ),
)
val x = a.lu().solve(doubleArrayOf(3.0, 5.0))

// Symmetric positive-definite solve via Cholesky.
val l = a.cholesky() // A = L·Lᵀ
val xs = solveSpd(l, doubleArrayOf(3.0, 5.0))

// Sparse solve: CSC matrix, sparse LU, FTRAN.
val sparse = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0)))
val xsp = SparseLu.factorize(sparse)!!.ftran(doubleArrayOf(3.0, 5.0))
```

## BLAS coverage

koblas implements a **well-defined subset** of double-precision BLAS/LAPACK — the kernels a revised
simplex or Bayesian-filtering workload needs — rather than "whatever the first consumer used". The
mapping, including deliberate deviations:

| BLAS / LAPACK | koblas | Notes |
|---|---|---|
| ddot | `dot` | sparse-aware |
| daxpy | `axpy` | sparse-aware x |
| dscal | `scale` | |
| dnrm2 | `norm2` | no overflow rescale: components within ~1e±150 |
| dasum | `asum` | |
| idamax | `iamax` | `-1` for zero length; storage-order ties for sparse |
| dcopy / dswap | `copy` / `swap` | |
| dgemv | `LinearAlgebra.gemv` | full alpha/beta/transpose form + fresh-result convenience |
| dger | `addOuter` | sparse-aware |
| dtrsv / dtrsm | `trsv` / `trsm` | trsm is left-side only; diagonal unchecked (as BLAS) |
| dgemm | `LinearAlgebra.gemm` | full alpha/beta + both transpose flags |
| dsyrk | `LinearAlgebra.syrk` | no `uplo`: full symmetric result, alpha term exactly symmetric |
| dgetrf / dgetrs | `factor` / `solve` | single right-hand side (use `trsm` for blocks) |
| — | `LuDecomposition.determinant` | 0.0 when singular; pairs with `SparseLu.determinant` |
| dpotrf | `cholesky` | non-standard `regularizeNonPD = true` default; pass `false` for strict |
| dpotrs / dpotri | `solveSpd` / `invertSpd` | |
| dchud / dchdd (LINPACK) | `choleskyUpdateInPlace` / `choleskyDowndateInPlace` | downdate reports cone exit instead of throwing |

Per BLAS convention, every alpha/beta form treats `beta == 0.0` as overwrite-without-reading (the
output may be uninitialized) and `alpha == 0.0` as the beta scale alone.

**Out of scope** (deliberately): banded/packed/triangular-packed layouts, single precision, complex,
`side = R` trsm, QR/SVD/eigendecompositions, and level-2 routines without a consumer here (symv,
trmv, ger2, …). If a workload needs one of these, it gets added to the table — nothing is supported
silently.

**Native backend contract:** any `platformLinearAlgebra()` implementation must match
`ReferenceLinearAlgebra` on the `BlasConformanceTest` suite, which validates every backend operation
(gemv, gemm, syrk, factor/solve, and the Cholesky family) against constructed references on standard
test matrices (identity, diagonal, Hilbert, random well-conditioned, SPD) within
theoretically-informed error bounds.

## Backends

Two seams, by granularity:

- **Level-1 SIMD primitives** (`dot`/`axpy`/`scale`) dispatch at compile time via `expect`/`actual`. The
  JVM uses the incubator Vector API (`jdk.incubator.vector`) when present, scalar loops otherwise; every
  other target is scalar. `mathBackend` reports the resolved kernel (e.g. `"simd(8 lanes)"` or `"scalar"`).
- **The `LinearAlgebra` backend** (`gemv`/`gemm`/LU) is runtime-swappable: `koblas` resolves to
  `platformLinearAlgebra()` when a target provides a native backend, else the portable
  `ReferenceLinearAlgebra`. A tuned BLAS/LAPACK or GPU backend slots in here without changing callers.

Storage is flat, contiguous, row-major — passable to a native backend as a raw buffer with no repacking.

JVM consumers that want the SIMD path pass `--add-modules=jdk.incubator.vector` at runtime; correctness
does not depend on it.

## Coordinates

```kotlin
implementation("com.eignex:koblas:<version>")
```

## Building

```
./gradlew check lintDocs
```

## Status

Early. The operation set is intentionally minimal — the pieces its consumers need — and grows on demand.
Native BLAS/LAPACK and GPU backends are planned behind the `LinearAlgebra` seam.
