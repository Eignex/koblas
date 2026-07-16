# koblas

Dense and sparse linear algebra for Kotlin Multiplatform.

Kotlin has no standard multiplatform BLAS/linear-algebra library. koblas is a small, focused one:
serializable matrix and vector containers, free-function arithmetic over them, and a per-platform
compute backend that a tuned BLAS/LAPACK (and, later, GPU) implementation can replace without touching
callers.

## What it provides

Containers — a read-only `View` contract, concrete backings that also expose their storage for in-place
work:

- `MatrixView` / `DenseMatrix` — flat, row-major `DoubleArray` backing.
- `VectorView` / `DenseVector` / `SparseVector` — dense or compressed-sparse.

Arithmetic — free functions over the views:

- BLAS-1/2: `dot`, `axpy`, `scale`, `addOuter` (rank-1 update), `matVec`, `forEachStored`.
- Symmetric positive-definite: `cholesky`, `choleskyDowndateInPlace`, `solveSpd`, `invertSpd`.

All containers are `@Serializable` (kotlinx.serialization); `DenseMatrix` serialises to a 2D
`Array<DoubleArray>` wire form independent of its flat backing.

## Usage

```kotlin
import com.eignex.koblas.*

val a = DenseMatrix.of(
    arrayOf(
        doubleArrayOf(2.0, 1.0),
        doubleArrayOf(1.0, 3.0),
    ),
)
val b = doubleArrayOf(3.0, 5.0)

val l = a.cholesky()   // A = L·Lᵀ
val x = solveSpd(l, b) // solves A·x = b
```

## Backends

Dense hot paths route through an `expect`/`actual` primitive seam. The JVM backend uses the incubator
Vector API (`jdk.incubator.vector`) when present and falls back to scalar loops; every other target is
scalar. `mathBackend` reports the resolved backend (e.g. `"simd(8 lanes)"` or `"scalar"`). Native
BLAS/LAPACK and GPU backends are planned behind the same seam.

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
