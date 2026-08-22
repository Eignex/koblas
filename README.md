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

Kotlin Multiplatform dense and sparse double-precision linear algebra. Koblas
provides BLAS/LAPACK operations, dense and sparse factorizations, portable
fallbacks, and optional OpenBLAS and SuiteSparse UMFPACK acceleration.

## Install

| Module | Purpose |
|--------|---------|
| koblas | Core API and portable backend. |
| koblas-openblas | Optional JVM bundle of OpenBLAS/LAPACKE for Linux x64/arm64 and macOS arm64. |
| koblas-umfpack | Optional GPL-3.0-only JVM bundle of SuiteSparse UMFPACK; includes koblas-openblas. |

```kotlin
implementation("com.eignex:koblas:<version>")
```

For host-native acceleration, install OpenBLAS/LAPACKE and, optionally,
SuiteSparse through your normal package manager (`apt`, `brew`, and so on).
Koblas discovers them automatically; without them it uses the portable backend.

On JVM Linux x64/arm64 and macOS arm64, Maven bundles are an alternative:

```kotlin
runtimeOnly("com.eignex:koblas-openblas:<version>")
runtimeOnly("com.eignex:koblas-umfpack:<version>") // optional; GPL-3.0-only
```

The UMFPACK bundle brings OpenBLAS and uses the same OpenBLAS library as the
dense backend. Bundled providers win over host lookup. To select a custom
absolute library path, use these JVM properties or environment variables:

| Library | JVM property | Environment variable |
|---------|--------------|----------------------|
| OpenBLAS | `koblas.openblas.path` | `KOBLAS_OPENBLAS_PATH` |
| LAPACKE | `koblas.lapacke.path` | `KOBLAS_LAPACKE_PATH` |
| UMFPACK | `koblas.umfpack.path` | `KOBLAS_UMFPACK_PATH` |

The JVM property takes precedence.

## Use

Containers use column-major storage. The unqualified names are aliases for the
implemented double-precision types, such as `F64DenseMatrix`.

```kotlin
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.solve

val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
val x = a.lu().solve(doubleArrayOf(3.0, 5.0))
println(x.contentToString()) // [0.8, 1.4]
```

Sparse matrices use CSC storage and can be created from readable columns or
coordinate triplets:

```kotlin
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.lu

val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))
val x = a.lu().solve(doubleArrayOf(4.0, 9.0))
```

The dense API covers level 1-3 BLAS, LU, Cholesky, QR, pivoted QR, LDLᵀ,
condition estimates, and inverses. Sparse LU and LDLᵀ are also available.

## Backends

Koblas starts with portable implementations and registers available accelerated
backends. Use `koblasInfo`, `koblas.portableSlots`, or
`koblas.requireAccelerated(...)` when acceleration is required rather than
optional. Set `-Dkoblas.backend=reference` to force the portable backend.

Host and bundled libraries are optional. A missing or unusable native library
does not stop the application; the affected operation falls back to the
portable implementation.

## License

The core is Apache-2.0. koblas-openblas also distributes BSD-3-Clause OpenBLAS
binaries. koblas-umfpack distributes GPL-3.0-only SuiteSparse UMFPACK in a
separate optional artifact.
