# koblas

Dense and sparse linear algebra for Kotlin Multiplatform, with a pluggable BLAS/LAPACK backend seam.

Kotlin has no standard multiplatform linear-algebra library. koblas is a small, focused one: a minimal
operation set (the pieces a dense linear solver actually needs) behind an interface, with a portable
pure-Kotlin backend that works on every target and a seam for native BLAS/LAPACK (and, later, GPU)
backends to slot in without changing consumers.

## What it provides

- `Matrix` — a flat, row-major dense matrix (BLAS/LAPACK-friendly layout).
- `Vector` — a thin dense-vector wrapper.
- `LinearAlgebra` — the operation set:
  - BLAS-1: `dot`, `axpy`, `scal`
  - BLAS-2/3: `gemv`, `gemm`
  - LU: `factor` (partial-pivot LU) and `solve`
- `ReferenceLinearAlgebra` — the portable pure-Kotlin backend, correct on every target.
- `platformLinearAlgebra()` — the per-platform native-backend seam (returns `null` today → reference).
- `koblas` — the resolved default backend (native if present, else reference).

## Usage

```kotlin
import com.eignex.koblas.*

val a = Matrix.ofRows(
    doubleArrayOf(2.0, 1.0),
    doubleArrayOf(1.0, 3.0),
)
val b = doubleArrayOf(3.0, 5.0)

val x = koblas.solve(koblas.factor(a), b) // solves A x = b
```

## Status

Early. The op set is intentionally minimal and grows as consumers demand. The reference backend is the
semantic reference; native BLAS/LAPACK backends (JVM via FFM→OpenBLAS/MKL, native via cinterop) and a GPU
backend are planned but not yet wired — every platform currently resolves to the reference backend.

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
