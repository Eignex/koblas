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

Dense and sparse double-precision linear algebra for Kotlin Multiplatform, with built-in C and JVM SIMD kernels.
Koblas provides BLAS operations, mutable matrices and vectors, strided views, and reusable workspaces.
See the [benchmark guide](koblas-bench/README.md) for performance suites and development-only OpenBLAS and
oneMKL comparisons. We welcome benchmark reports from different hardware.

## Setup

Add the Maven Central dependency to your JVM or shared `commonMain` source set, replacing `<version>` with the
version shown in the badge above:

```kotlin
implementation("com.eignex:koblas:<version>")
```

Supported targets are JVM (JDK 25 or later), Linux x64/arm64, and macOS arm64.

On JVM, enable SIMD with the runtime flag `--add-modules=jdk.incubator.vector`. Without it, Koblas uses bundled
C kernels when available, then scalar Kotlin. For bundled C kernels on the classpath, also pass
`--enable-native-access=ALL-UNNAMED`. Kotlin/Native uses bundled C kernels with scalar fallbacks.

## Quick start

```kotlin
import com.eignex.koblas.*
import com.eignex.koblas.dense.*

val a = DenseMatrix.of(arrayOf(
    doubleArrayOf(2.0, 1.0),
    doubleArrayOf(1.0, 3.0),
))
val b = DenseMatrix.diagonal(2)
val x = DenseVector.of(doubleArrayOf(3.0, 5.0))

val product = a * b
val y = a * x
```

Use operators for allocating arithmetic, or BLAS-style and `*Into` operations to write into existing buffers.
A `Workspace` grows as needed and retains temporary storage for repeated calls:

```kotlin
val input = DoubleArray(a.rows)
val output = DoubleArray(a.cols)
val workspace = Workspace()

repeat(1_000) {
    koblas.gemv(1.0, a, input, 0.0, output, transpose = true, workspace = workspace)
}
```

## Operations

| Area | Coverage |
|------|----------|
| Dense vectors | Dot products, scaling, norms, copy, swap, and rotations. |
| Dense matrices | General, symmetric, and triangular products; triangular solves; rank updates; transpose and norms. |
| Sparse vectors | Indexed dot products, scaled addition, scatter, and norms. |
| Sparse matrices | CSC products, symmetric products, triangular products and solves, rank updates, addition, and transpose. |

Sparse matrix products can produce sparse or dense results. For repeated products, `SparseMatrix.prepare()`
(from `com.eignex.koblas.sparse`) retains an immutable CSC snapshot.

See the [dense](koblas/src/commonMain/kotlin/com/eignex/koblas/dense/Blas.kt) and
[sparse](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparseBlas.kt) API contracts for overloads and
supported routines.

## Storage and reuse

All containers use `Double` values. `DenseMatrix` stores columns contiguously: `A(i, j)` is at `i + j * rows`.
`SparseMatrix` uses validated CSC storage with `Int` row indices in ascending order within each column.
Compatible arrays can be wrapped without copying.

Dense views borrow live storage and preserve offsets, strides, and leading dimensions:

```kotlin
val storage = DenseMatrix.zero(512, 32)
val panel = storage.view(row = 64, rows = 128, column = 4, cols = 8)
val weights = DenseMatrix.zero(8, 2)
val result = DenseMatrix.zero(128, 2)

koblas.gemm(
    alpha = 1.0,
    a = panel,
    transposeA = false,
    b = weights.asView(),
    transposeB = false,
    beta = 0.0,
    c = result.asView(),
)
```

Follow each operation's aliasing contract. Disjoint views can share a backing buffer, but a strided destination
must not overlap an input. Symmetric sparse operations read only the selected triangle; use `symv` or `symm`
to interpret a single-triangle result symmetrically.

The default `koblas` engine is immutable, selected once, and safe to share. Its kernels are single-threaded.
Containers and views are mutable; concurrent reads require no writer. Use one `Workspace` per concurrent
operation.
