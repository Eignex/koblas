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

Koblas provides dense and sparse double-precision linear algebra for Kotlin Multiplatform. It includes BLAS
operations, mutable matrices and vectors, views into existing storage, and reusable workspaces. Built-in C and
JVM SIMD kernels speed up supported operations.

SIMD is confined to the low-level kernels. Level 2 and Level 3 operations are implemented in portable Kotlin
and call those kernels to accelerate their inner loops.

See the [benchmark guide](koblas-bench/README.md) for performance tests and comparisons with OpenBLAS and
oneMKL. We welcome benchmark reports from different hardware.

## Setup

Add the Maven Central dependency to your JVM or `commonMain` source set. Replace `<version>` with the version
shown in the badge above:

```kotlin
implementation("com.eignex:koblas:<version>")
```

Supported targets are JVM (JDK 25 or later), Linux x64/arm64, and macOS arm64. When the Vector API is made stable
that will be the new JVM lowest target.

On JVM, pass `--add-modules=jdk.incubator.vector` at runtime to enable SIMD. Without this flag, Koblas uses its
bundled C engine when available. This engine runs small operations in scalar Kotlin and switches to C for larger
ones. If the C engine is unavailable, Koblas uses scalar Kotlin for all operations.

To load the bundled C kernels from the classpath, also pass `--enable-native-access=ALL-UNNAMED`. Kotlin/Native
uses bundled C kernels and falls back to scalar Kotlin when needed.

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

Operators create a new result. BLAS-style and `*Into` operations can instead write into existing arrays and
containers. A `Workspace` keeps temporary storage for reuse across calls:

```kotlin
val input = DoubleArray(a.rows)
val output = DoubleArray(a.cols)
val workspace = Workspace()

repeat(1_000) {
    koblas.gemv(1.0, a, input, 0.0, output, transpose = true, workspace = workspace)
}
```

### Global and explicit contexts

Operators such as `a * b` use the global, platform-selected `koblas` context. Pass a `KoblasContext` when the
caller should choose the engine:

```kotlin
fun multiply(context: KoblasContext, left: DenseMatrix, right: DenseMatrix): DenseMatrix =
    context.gemm(left, right)

val scalarProduct = multiply(BuiltinEngines.scalar, a, b)
```

## Operations

| API | Includes |
|-----|----------|
| [Dense matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/Matrix.kt), [sparse matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/SparseMatrix.kt), [vectors](koblas/src/commonMain/kotlin/com/eignex/koblas/Vector.kt), and [views](koblas/src/commonMain/kotlin/com/eignex/koblas/StridedViews.kt) | Dense and sparse containers, array wrapping, factories, and views into existing storage. |
| [Vector operations](koblas/src/commonMain/kotlin/com/eignex/koblas/VectorOps.kt) | Dense and sparse dot products, sums, norms, scaling, copy, swap, gather, scatter, and [Givens](koblas/src/commonMain/kotlin/com/eignex/koblas/Givens.kt) or [modified Givens](koblas/src/commonMain/kotlin/com/eignex/koblas/ModifiedGivens.kt) rotations. |
| [Matrix helpers](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixOps.kt) | Allocating [operators](koblas/src/commonMain/kotlin/com/eignex/koblas/Operators.kt), matrix-vector products, rank updates, [slices and transpose](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixSlices.kt), [scaling](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixScaling.kt), and [norms](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixNorms.kt). |
| [Dense BLAS](koblas/src/commonMain/kotlin/com/eignex/koblas/dense/Blas.kt) | General, symmetric, and triangular matrix products and solves, including `gemmt`, `syr2k`, and strided-view overloads. |
| [Sparse BLAS](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparseBlas.kt) | Sparse matrix-vector and matrix-matrix products, dense or sparse results, symmetric and triangular operations, addition, and transpose. |
| [Packed panels](koblas/src/commonMain/kotlin/com/eignex/koblas/dense/PackedPanels.kt) and [tile kernels](koblas/src/commonMain/kotlin/com/eignex/koblas/dense/PackedKernels.kt) | Reusable packed layouts and fixed-size product and triangular-solve tiles for custom blocked algorithms. |
| [Sparse kernels](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparseKernels.kt) and [slices](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparseSlices.kt) | Allocation-free indexed arithmetic over caller-owned arrays, including accumulation, touched-index handling, diagnostics, and pivot candidates. |
| [Engines and kernel APIs](koblas/src/commonMain/kotlin/com/eignex/koblas/KoblasContext.kt) | The default engine, explicit scalar, C, and SIMD engines, and lower-level dense, packed, and sparse kernel interfaces. |

Sparse matrix products can produce sparse or dense results. If you reuse a sparse matrix in several products,
call `SparseMatrix.prepare()` from `com.eignex.koblas.sparse` once and reuse the prepared copy.

## Error handling

Shape mismatches throw `DimensionMismatch`, invalid logical indices throw `IndexOutOfBoundsException`, and other
invalid arguments throw `IllegalArgumentException`. Numerical failures use `KoblasException` subtypes.

## Storage and reuse

All matrices and vectors use `Double` values. `DenseMatrix` uses column-major storage, so `A(i, j)` is at
`i + j * rows`. `SparseMatrix` uses compressed sparse column (CSC) storage with sorted `Int` row indices.
Compatible arrays can be wrapped without copying them.

Dense views refer to the original storage instead of copying it. They can represent part of a matrix or vector:

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

Inputs and outputs may share an array only when the operation's documentation allows it. Two views can use the
same array if they do not cover the same elements. An output view that skips positions in the array must not
overlap an input.

For symmetric sparse operations, Koblas reads only the triangle you select. Use `symv` or `symm` to treat that
triangle as a full symmetric matrix.

Koblas chooses its default engine once, and the engine is safe to share. Its kernels use one thread. Matrices,
vectors, and views are mutable, so do not modify them while another thread is reading them. Give each operation
running at the same time its own `Workspace`.
