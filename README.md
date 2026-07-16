# koblas

Dense and sparse linear algebra for Kotlin Multiplatform.

Kotlin has no standard multiplatform linear-algebra library. koblas is a small, focused one: read-only
matrix/vector containers with sealed dense/sparse backings, free-function arithmetic over them, and a
per-platform primitive backend (SIMD on the JVM today; a tuned BLAS/LAPACK — and later GPU — backend can
slot in behind the same seam).

## What it provides

- `MatrixView` / `DenseMatrix` — read-only matrix with a flat, row-major `DoubleArray` backing.
- `VectorView` / `DenseVector` / `SparseVector` — read-only vector, dense or compressed-sparse.
- Free-function arithmetic over the views:
  - BLAS-1/2: `dot`, `axpy`, `scale`, `addOuter` (rank-1 update), `matVec`, `forEachStored`.
  - SPD suite: `cholesky`, `choleskyDowndateInPlace`, `solveSpd`, `invertSpd`.
- `mathBackend` — the resolved primitive backend id (`"simd(N lanes)"` on the JVM when the incubator
  Vector API is available, else `"scalar"`).

All container types are `@Serializable` (kotlinx.serialization); the dense matrix serialises to a 2D
`Array<DoubleArray>` wire form independent of its flat backing.

## Usage

```kotlin
import com.eignex.koblas.*

val a = DenseMatrix.of(arrayOf(
    doubleArrayOf(2.0, 1.0),
    doubleArrayOf(1.0, 3.0),
))
val b = doubleArrayOf(3.0, 5.0)

val l = a.cholesky()        // A = L Lᵀ
val x = solveSpd(l, b)      // solves A x = b
```

## Status

Early. The op set is intentionally minimal — the pieces its consumers need — and grows on demand.
Dense hot paths use SIMD on the JVM (`jdk.incubator.vector`) with a scalar fallback everywhere; native
BLAS/LAPACK and GPU backends are planned behind the same primitive seam.

## Coordinates

```kotlin
implementation("com.eignex:koblas:<version>")
```

## Building

```
./gradlew check lintDocs
```

## License

See [LICENSE](LICENSE).
