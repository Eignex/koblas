<p align="center">
  <a href="https://eignex.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner-white.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg">
      <img alt="Eignex" src="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg" style="max-width: 100%; width: 22em;">
    </picture>
  </a>
</p>

# Koblas

[![Maven Central](https://img.shields.io/maven-central/v/com.eignex/koblas.svg?label=Maven%20Central)](https://central.sonatype.com/artifact/com.eignex/koblas)
[![Build](https://github.com/eignex/koblas/actions/workflows/build.yml/badge.svg)](https://github.com/eignex/koblas/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/eignex/koblas/branch/main/graph/badge.svg)](https://codecov.io/gh/eignex/koblas)
[![License](https://img.shields.io/github/license/eignex/koblas)](https://github.com/eignex/koblas/blob/main/LICENSE)

Dense and sparse double-precision linear algebra for Kotlin Multiplatform.
Koblas provides BLAS/LAPACK operations, factorizations, and optional OpenBLAS,
SuiteSparse KLU, or SuiteSparse UMFPACK acceleration.

## Install

| Module | Purpose |
|--------|---------|
| koblas | Core API and portable backend. |
| koblas-openblas | OpenBLAS/LAPACKE runtime bundle. |
| koblas-umfpack | UMFPACK sparse-LU runtime bundle. |
| koblas-klu | KLU sparse-LU runtime bundle. |
| koblas-basiclu | BASICLU sparse-basis runtime bundle. |

Install OpenBLAS/LAPACKE and, for sparse LU, SuiteSparse KLU 2 or UMFPACK with your
system package manager, such as `apt` or Homebrew. Koblas discovers them automatically
and otherwise uses its portable backend. On JVM Linux x64/arm64 and macOS arm64, the
optional bundled modules are an alternative; host packages and Maven bundles use the
same Koblas bindings, differing only in the native library source.

The UMFPACK bundle brings OpenBLAS and uses the same OpenBLAS library as the
dense backend; KLU and BASICLU have no BLAS dependency. Bundled providers win over host
lookup. Optional bundled modules have licenses in addition to Apache 2.0; see their
generated `THIRD-PARTY-NOTICES.txt` files for details.

## Use

Containers use column-major storage. Unqualified names such as `DenseMatrix`
are aliases for the implemented double-precision types (`F64DenseMatrix`).

```kotlin
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.solve

val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
val x = a.lu().solve(doubleArrayOf(3.0, 5.0))
println(x.contentToString()) // [0.8, 1.4]
```

For a symmetric positive-definite system, use Cholesky factorization:

```kotlin
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.solve

val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
val x = a.cholesky().solve(doubleArrayOf(3.0, 5.0))
```

For an overdetermined system, QR produces a least-squares solution:

```kotlin
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.qr
import com.eignex.koblas.dense.solveLeastSquares

val design = DenseMatrix.of(
    arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0, 1.0)),
)
val coefficients = design.qr().solveLeastSquares(doubleArrayOf(1.0, 2.0, 3.0))
```

Operator functions cover matrix arithmetic, products, and matrix-vector products:

```kotlin
import com.eignex.koblas.*

val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
val b = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(1.0, 2.0)))
val x = DenseVector.of(doubleArrayOf(2.0, -1.0))

val sum = a + b
val product = a * b
val y = a * x
```

Sparse matrices use CSC storage. Like dense matrices, they support vector operations,
matrix-vector products, and triangular solves. Their factorization is LU only, with
refactorization and simplex-basis column replacement for repeated solves and updates.
Construct them from columns or coordinate triplets:

```kotlin
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.lu

val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))
val x = a.lu().solve(doubleArrayOf(4.0, 9.0))
```

## Backends

Koblas starts with portable implementations. Configure accelerated backends either
programmatically or automatically.

For programmatic control, use `registerBackend(...)` to offer a backend, or
`installBackends(...)` to override the process-wide context. Programmatic
registrations take precedence over automatically discovered backends.

For automatic JVM configuration, call `discoverBackends()`. Koblas probes the host
and bundled libraries and falls back to the portable implementation when a library
cannot load. Use these JVM properties or environment variables to steer discovery
to custom absolute library paths:

| Library | JVM property | Environment variable |
|---------|--------------|----------------------|
| CBLAS | `koblas.cblas.path` | `KOBLAS_CBLAS_PATH` |
| LAPACKE | `koblas.lapacke.path` | `KOBLAS_LAPACKE_PATH` |
| SuiteSparse KLU 2 | `koblas.klu.path` | `KOBLAS_KLU_PATH` |
| UMFPACK | `koblas.umfpack.path` | `KOBLAS_UMFPACK_PATH` |

The JVM property takes precedence. `koblas.klu.path` must name a SuiteSparse KLU 2
library. Set `-Dkoblas.dense.backend=reference -Dkoblas.sparse.backend=reference`
to force portable backends. With both sparse-LU bundles present, set
`-Dkoblas.sparse.backend=umfpack` to select UMFPACK.

Inspect `koblasInfo` or `koblas.portableSlots` after configuration; call
`koblas.requireAccelerated(...)` to fail when acceleration is unavailable.
