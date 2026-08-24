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
| koblas-openblas | Optional JVM bundle of OpenBLAS/LAPACKE for Linux x64/arm64 and macOS arm64. |
| koblas-umfpack | Optional JVM bundle of SuiteSparse UMFPACK (GPL-2.0-or-later); includes koblas-openblas. |
| koblas-klu | Optional JVM bundle of SuiteSparse KLU (LGPL-2.1-or-later). |
| koblas-basiclu | Optional JVM bundle of BASICLU for sparse simplex-basis factorization and updates (MIT). |

All optional hosted modules have a bunch of different licenses in addition to Apache 2.0, see it's generated `THIRD-PARTY-NOTICES.txt` for specifics.

```kotlin
implementation("com.eignex:koblas:<version>")
```

Install OpenBLAS/LAPACKE and, for sparse LU, SuiteSparse KLU 2 or UMFPACK with your
system package manager, such as `apt` or Homebrew. Koblas discovers them automatically
and otherwise uses its portable backend.

On JVM Linux x64/arm64 and macOS arm64, Maven bundles are an alternative:

```kotlin
runtimeOnly("com.eignex:koblas-openblas:<version>")
runtimeOnly("com.eignex:koblas-klu:<version>") // optional, sparse LU default
runtimeOnly("com.eignex:koblas-umfpack:<version>") // optional alternative
runtimeOnly("com.eignex:koblas-basiclu:<version>") // optional, sparse simplex bases
```

Host packages and Maven bundles use the same Koblas bindings; only the native library
source differs.

The UMFPACK bundle brings OpenBLAS and uses the same OpenBLAS library as the
dense backend; KLU and BASICLU have no BLAS dependency. Bundled providers win over host lookup. To
select a custom absolute library path, use these JVM properties or environment variables:

| Library | JVM property | Environment variable |
|---------|--------------|----------------------|
| CBLAS | `koblas.cblas.path` | `KOBLAS_CBLAS_PATH` |
| LAPACKE | `koblas.lapacke.path` | `KOBLAS_LAPACKE_PATH` |
| SuiteSparse KLU 2 | `koblas.klu.path` | `KOBLAS_KLU_PATH` |
| UMFPACK | `koblas.umfpack.path` | `KOBLAS_UMFPACK_PATH` |

The JVM property takes precedence. `koblas.klu.path` must name a SuiteSparse KLU 2
library. With both sparse-LU bundles present, KLU is selected by default; set
`koblas.sparse.backend=umfpack` to select UMFPACK.

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

Sparse matrices use CSC storage. Construct them from columns or coordinate
triplets:

```kotlin
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.lu

val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))
val x = a.lu().solve(doubleArrayOf(4.0, 9.0))
```

The dense API covers level 1-3 BLAS, LU, Cholesky, QR, pivoted QR, LDLᵀ,
condition estimates, and inverses. Sparse LU is also available.

## Backends

Koblas starts with portable implementations and registers available accelerated
backends. Inspect `koblasInfo` or `koblas.portableSlots`; call
`koblas.requireAccelerated(...)` to fail when acceleration is unavailable. Set
`-Dkoblas.dense.backend=reference -Dkoblas.sparse.backend=reference` to force portable backends.

Host and bundled libraries are optional. If one cannot load, Koblas falls back
to the portable implementation.
