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
operations, mutable matrices and vectors, and views into existing storage.

Koblas splits at the BLAS level. Level 1 and the sparse primitives are portable Kotlin, with JVM SIMD kernels
where the Vector API is available, and they work on any host. Level 2 and Level 3 are whole calls to an
installed vendor BLAS — oneMKL, AOCL, or Arm Performance Libraries — each held to one compute thread. Koblas
validates shapes and aliasing there and contributes no arithmetic of its own. On a host with no supported
library the containers, Level 1 and the sparse primitives keep working; an accelerator-dependent call raises
rather than quietly computing something slower under the same name.

Koblas itself has no external dependencies and is entirely Apache-2.0 licensed.

The [benchmark guide](koblas-bench/README.md) times each arm against the others and against OpenBLAS and
oneMKL. We welcome benchmark reports from different hardware.

## Setup

Add the Maven Central dependency to your JVM or `commonMain` source set. Replace `<version>` with the version
shown in the badge above:

```kotlin
implementation("com.eignex:koblas:<version>")
```

Supported targets are JVM (JDK 25 or later), Linux x64/arm64, and macOS arm64. When the Vector API is made stable
that will be the new JVM lowest target.

On JVM, pass `--add-modules=jdk.incubator.vector` at runtime to enable the SIMD Level 1 kernels. Without it
Koblas uses portable Kotlin, which is also what the SIMD kernels fall back to below their lane width and for
any strided run.

Level 2 and Level 3 need an installed vendor BLAS. Koblas looks for oneMKL, AOCL and Arm Performance Libraries
by their usual file names and the prefixes their installers use, so no environment setup is required; the
resolved file is reported back so a run says which library actually ran. On JVM those calls are foreign
downcalls, so pass `--enable-native-access=ALL-UNNAMED`.

## Quick start

```kotlin
import com.eignex.koblas.*

val a = DenseMatrix.ofRows(arrayOf(
    doubleArrayOf(2.0, 1.0),
    doubleArrayOf(1.0, 3.0),
))
val b = DenseMatrix.diagonal(2)
val x = DenseVector.of(doubleArrayOf(3.0, 5.0))

val product = a * b
val y = a * x
```

Operators create a new result. BLAS-style and `*Into` operations instead write into arrays the caller already
owns, so a loop allocates nothing of its own:

```kotlin
val input = DoubleArray(a.rows)
val output = DoubleArray(a.cols)

repeat(1_000) {
    koblas.gemv(1.0, a, input, 0.0, output, transpose = true)
}
```

## Java

The JVM artifact exposes container factories as static methods and collects Kotlin's operators and extensions on
the `Koblas` class directly from their canonical declarations. Default arguments have Java overloads, so common
calls do not need `Companion`, placeholder flags, or generated `*Kt` class names:

```java
import com.eignex.koblas.DenseMatrix;
import com.eignex.koblas.DenseVector;
import com.eignex.koblas.Koblas;
import com.eignex.koblas.SparseMatrix;

DenseMatrix a = DenseMatrix.ofRows(new double[][] {
    {2.0, 1.0},
    {1.0, 3.0},
});
DenseVector x = DenseVector.of(new double[] {3.0, 5.0});

DenseVector y = Koblas.multiply(a, x);
double norm = Koblas.norm2(x);

double[] destination = new double[a.getRows()];
Koblas.gemvInto(a, x, destination);

SparseMatrix diagonal = SparseMatrix.ofTriplets(
    2,
    2,
    new int[] {0, 1},
    new int[] {0, 1},
    new double[] {2.0, 3.0}
);
```

`DenseMatrix`, `DenseVector`, `SparseMatrix`, and `SparseVector` provide static `of`, `ofRows`, `ofTriplets`,
`zero`, `diagonal`, and `wrap` factories as appropriate. Values use ordinary Java primitive arrays. Read entries
with `get(...)`, mutate dense storage with `set(...)`, and use `getData()` or `getValues()` when direct mutable
storage access is intentional. `Koblas.getDefault()` exposes the selected engine.

### The selected engine

Operators such as `a * b` use the global, platform-selected `koblas` engine, which is chosen once and immutable.
Pass a `KoblasEngine` explicitly when a caller should be able to supply one:

```kotlin
fun multiply(engine: KoblasEngine, left: DenseMatrix, right: DenseMatrix): DenseMatrix =
    engine.gemm(left, right)

val product = multiply(koblas, a, b)
```

Naming a Level 1 implementation other than the selected one is for measuring the two against each other, so
`BuiltinEngines` sits behind the `KoblasEngineApi` opt-in. The portable kernels are not an alternative to the
SIMD ones at a given size: the SIMD ones already fall back to them below their lane width and for any strided
run. `KoblasEngine.explain(operation, length, contiguous)` names the implementation a given call reaches.

## API structure

Most application code needs only `com.eignex.koblas.*`. The root package contains the owning dense and sparse
containers, zero-copy vector views, allocating operators, and high-level operations. `Matrix` and `Vector` are
read-only contracts that custom types can implement. `MatrixStorage` and `VectorStorage` identify Koblas's
built-in dense and sparse containers, so storage-level operations can use one API and dispatch according to the
actual storage. `DenseVector` is sealed over the two dense spacings, adjacent and strided, because a vendor
takes either as a pointer and an increment; `SparseVector` is deliberately not one, so it does not compile where
a dense operand is wanted. Matrix arithmetic takes dense storage.

The `com.eignex.koblas.dense`, `com.eignex.koblas.sparse` and `com.eignex.koblas.vendor` packages are the
lower-level composition layer. `DenseBlas` exposes Koblas's dense BLAS signatures and `Blas` the standard's own
argument shapes underneath them, while the kernel interfaces and sparse primitives support custom algorithms
over caller-owned storage. Ordinary matrix and vector arithmetic does not require imports from these packages.

`KoblasEngine` connects the layers: it binds the portable Level 1 and sparse kernels to the selected vendor and
implements the dense BLAS contract on top. Root-package operators and extensions use the platform-selected
`koblas` engine.

## Operations

| API | Includes |
|-----|----------|
| [Dense matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/Matrix.kt), [sparse matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/SparseMatrix.kt), [vectors](koblas/src/commonMain/kotlin/com/eignex/koblas/Vector.kt), and [strided vectors](koblas/src/commonMain/kotlin/com/eignex/koblas/StridedVector.kt) | Dense and sparse containers, array wrapping, factories, and borrowed views into existing vector storage. |
| [Vector operations](koblas/src/commonMain/kotlin/com/eignex/koblas/VectorOps.kt) | Dense and sparse dot products, sums, norms, scaling, copy, swap, gather, scatter, and [plane rotations](koblas/src/commonMain/kotlin/com/eignex/koblas/Rot.kt). |
| [Matrix helpers](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixOps.kt) | Allocating [operators](koblas/src/commonMain/kotlin/com/eignex/koblas/Operators.kt), matrix-vector products, rank updates, [triangular operations](koblas/src/commonMain/kotlin/com/eignex/koblas/Triangular.kt), [slices](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixSlices.kt), [scaling and masking](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixScaling.kt), and transpose. |
| [Dense BLAS](koblas/src/commonMain/kotlin/com/eignex/koblas/dense/DenseBlas.kt) | General, symmetric, and triangular matrix products and solves, including `gemmt` and `syr2k`, served by the vendor. |
| [Vendor binding](koblas/src/commonMain/kotlin/com/eignex/koblas/vendor/Blas.kt) | The CBLAS entry points in the standard's own argument shapes, library identity, single-thread evidence, and the route a concrete call takes. |
| [Sparse kernels](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparseKernels.kt) and [primitives](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparsePrimitives.kt) | Allocation-free indexed arithmetic over caller-owned arrays, including accumulation, touched-index handling, diagnostics, and pivot candidates. |
| [Engine](koblas/src/commonMain/kotlin/com/eignex/koblas/KoblasEngine.kt) | The selected engine, its Level 1 attribution, and the opt-in seam for naming an implementation to measure. |

Sparse support is storage, Level 1, and the generic primitives. A consumer that needs sparse products or
factorizations builds them on `SparsePrimitives` and owns its own factors.

## Error handling

Shape mismatches throw `DimensionMismatch`, invalid logical indices throw `IndexOutOfBoundsException`, and other
invalid arguments throw `IllegalArgumentException`. Numerical failures use `KoblasException` subtypes.

## Storage and reuse

All matrices and vectors use `Double` values. `DenseMatrix` uses column-major storage, so `A(i, j)` is at
`i + j * rows`. `SparseMatrix` uses compressed sparse column (CSC) storage with sorted `Int` row indices.
Compatible arrays can be wrapped without copying them.

A `StridedVector` refers to the original storage instead of copying it, and can address part of an array or a
row or column of a matrix. Every dense routine takes one wherever it takes a contiguous vector, because the
spacing is an increment the library is handed rather than a different algorithm:

```kotlin
val storage = DenseMatrix.zero(512, 32)
val firstColumn = storage.column(0)
val everySecondEntry = DenseVector.wrap(DoubleArray(64)).view(offset = 0, size = 32, stride = 2)

firstColumn.scale(2.0)
val alignment = firstColumn dot everySecondEntry
```

Matrices are passed whole. Level 2 and Level 3 take a `DenseMatrix` with its transpose and stored triangle
travelling beside it as arguments, so there is no submatrix type to construct.

Inputs and outputs may share an array only when the operation's documentation allows it. The rank updates and
matrix products refuse a destination that shares a buffer with an input, because the standard leaves that
undefined; the triangular solves write their destination in place by design.

For symmetric operations, Koblas reads only the triangle you select. Use `symv` or `symm` to treat that
triangle as a full symmetric matrix.

Koblas chooses its default engine once, and the engine is safe to share. Every vendor call runs on one compute
thread, established when the library is opened and not configurable. Matrices, vectors, and views are mutable,
so do not modify them while another thread is reading them.
