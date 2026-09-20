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

Koblas provides dense and sparse double-precision BLAS Levels 1–3 for Kotlin Multiplatform.
Portable Kotlin implementations work without an installed numerical library. JVM performance uses owned
Vector API kernels; Kotlin/Native can accelerate eligible calls through installed host libraries.
Koblas has no external runtime dependencies and is Apache-2.0 licensed. It bundles no vendor libraries or
custom native numerical kernels.

## Setup

Add the Maven Central dependency to your JVM or `commonMain` source set:

```kotlin
implementation("com.eignex:koblas:<version>")
```

Supported targets: JVM on JDK 25 or later, Linux x64/arm64, and macOS arm64.

- JVM: pass `--add-modules=jdk.incubator.vector` to enable the SIMD engine. Without it, calls use portable
  Kotlin. Ordinary JVM calls do not load a host library.
- Kotlin/Native: eligible calls use an installed host library when available, with portable fallback.
- Explicit JVM host bindings also need `--enable-native-access=ALL-UNNAMED`. Supported vendors include
  OpenBLAS, oneMKL, AOCL, Arm Performance Libraries and Accelerate.

The immutable `koblas` engine is selected once. An operation can combine SIMD, portable and host components;
selection alone does not identify what executed. `explain`, `denseRouteOf` and `SparseBlas.matrixRouteOf`
report actual execution. Host bindings report the resolved binary, version and available thread evidence.
`BuiltinEngines`, behind the `KoblasEngineApi` opt-in, names exact engines for tests and benchmarks.

## Kotlin

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

Operators allocate results. BLAS-style and `*Into` calls write into caller-owned destinations; supply a
`Workspace` where supported to reuse staging and packing scratch:

```kotlin
val result = DenseMatrix.zero(2, 2)
val workspace = Workspace()

repeat(1_000) {
    koblas.gemm(1.0, a, false, b, false, 0.0, result, workspace)
}
```

A workspace retains a bounded set of buffers. It does not eliminate allocations for fresh results or every
convenience operation. Concurrent calls need distinct destinations and workspaces.

## Java

Factories are static methods, and Kotlin operators/extensions are exposed through `Koblas`. Common calls
have overloads for default arguments:

```java
import com.eignex.koblas.DenseMatrix;
import com.eignex.koblas.DenseVector;
import com.eignex.koblas.Koblas;

DenseMatrix a = DenseMatrix.ofRows(new double[][] {{2.0, 1.0}, {1.0, 3.0}});
DenseVector x = DenseVector.of(new double[] {3.0, 5.0});
DenseVector y = Koblas.multiply(a, x);

double[] destination = new double[a.getRows()];
Koblas.gemvInto(a, x, destination);
```

Use `Koblas.getDefault()` for the selected engine. Containers expose ordinary primitive arrays when direct
storage access is needed.

## Storage and contracts

- `DenseMatrix` is column-major: `A(i, j)` is stored at `i + j * rows`.
- `SparseMatrix` is validated CSC with ascending row indices. Stored zeros participate in arithmetic;
  absent entries are not evaluated, and cancellation does not implicitly remove stored entries.
- `Matrix` products accept dense or sparse operands in either position, including independent transpose
  flags. Two sparse operands produce a fresh CSC result; other built-in pairings produce dense results.
  `gemmInto` writes any pairing into a dense destination without densifying sparse inputs.
- Arrays can be wrapped without copying. `StridedVector` and matrix row/column views borrow existing storage.
- Built-in `*Into` paths support aliases through snapshotting. Explicit vendor bindings retain their own
  overlap contract. Symmetric operations read only the selected triangle; triangular solves write in place.
- Prepared sparse matrices own their snapshots and can be shared by readers with separate mutable scratch.
- Each call uses one compute thread. Defaults are immutable; containers, views and workspaces are mutable.

Shape mismatches throw `DimensionMismatch`; invalid indices throw `IndexOutOfBoundsException`; other invalid
arguments throw `IllegalArgumentException`. Numerical failures use `KoblasException` subtypes. Behavior left
open by BLAS, such as tie/NaN choices in `idamax`, follows the selected implementation.

## API guide

Most application code needs only `com.eignex.koblas.*`. Lower-level interfaces support custom algorithms over
caller-owned storage. Factorization and solver workflows belong to consumers.

| Area | API |
|---|---|
| Containers and views | [Matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/Matrix.kt), [sparse matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/SparseMatrix.kt), [vectors](koblas/src/commonMain/kotlin/com/eignex/koblas/Vector.kt) |
| Vector arithmetic | [VectorOps](koblas/src/commonMain/kotlin/com/eignex/koblas/VectorOps.kt) |
| Matrix operations | [MatrixOps](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixOps.kt), [generic products](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixProducts.kt) |
| BLAS | [DenseBlas](koblas/src/commonMain/kotlin/com/eignex/koblas/dense/DenseBlas.kt), [SparseBlas](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparseBlas.kt) |
| Reuse | [Workspace](koblas/src/commonMain/kotlin/com/eignex/koblas/Workspace.kt), [prepared sparse snapshots](koblas/src/commonMain/kotlin/com/eignex/koblas/PreparedSparseMatrix.kt) |
| Custom kernels and bindings | [SparsePrimitives](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparsePrimitives.kt), [vendor Blas](koblas/src/commonMain/kotlin/com/eignex/koblas/vendor/Blas.kt) |

See the [API overview](koblas/module.md) and [benchmark guide](koblas-bench/README.md) for kernel contracts,
exact-engine comparisons and reproducible captures.

## Development

Run `./gradlew check lintDocs`. Add `-Pkoblas.noSimd=true` to check the JVM fallback without the Vector API
module; this does not hide installed vendor libraries.
