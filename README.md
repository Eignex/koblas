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
SuiteSparse KLU, SuiteSparse UMFPACK, or BASICLU acceleration.

Koblas is for numerical and optimization routines that need predictable performance
and precise control over matrix storage, allocations, and workspaces. It is especially
suited to repeated solves and refactorizations. Typical uses include vector embeddings
and similarity search, simplex and other optimization methods, statistical estimation,
regression, and custom scientific-computing algorithms. BF16 support is planned for
embedding workloads.

Unlike [Kotlin DataFrame](https://github.com/Kotlin/dataframe), Koblas is a low-level
linear-algebra building block for applications that already own their numerical data
and algorithms. Its API exposes dense and sparse storage, BLAS/LAPACK-style
operations, and caller-managed outputs and workspaces so compatible `DoubleArray` and
CSC buffers can be wrapped without copying and hot paths can avoid allocations where
possible.

## Install

| Module | Purpose |
|--------|---------|
| koblas | Dense and sparse API with a portable backend. |
| koblas-openblas | Accelerated dense BLAS and LAPACK operations. |
| koblas-umfpack | General sparse LU for irregular or unstructured systems. |
| koblas-klu | Sparse LU for circuit-style systems with a fixed sparsity pattern. |
| koblas-basiclu | Simplex-basis LU with efficient column replacements. |
| koblas-hfactor | Simplex-basis solver with hypersparse solves and Forrest-Tomlin updates. |

On JVM, add `--add-modules=jdk.incubator.vector` to enable the built-in SIMD kernels.
They beat the native binding for BLAS level 1 (vector-vector work) and level 2
(matrix-vector work), so Koblas keeps them on the JVM and reserves the native backend
for level 3 matrix-matrix operations and factorizations.

Install OpenBLAS/LAPACKE for dense acceleration and SuiteSparse KLU 2 or UMFPACK for
sparse LU with a system package manager such as apt or Homebrew. Koblas discovers host
libraries automatically and falls back to its portable backend when they are unavailable.

On JVM Linux x64/arm64 and macOS arm64, optional bundled modules take precedence over
host libraries. `koblas-umfpack` bundles OpenBLAS for dense operations; KLU and BASICLU
do not need it. Bundled modules have licenses in addition to Apache 2.0; see their
generated `THIRD-PARTY-NOTICES.txt` files for details.

## Use

Containers use column-major storage. Unqualified names such as `DenseMatrix`
are aliases for the implemented double-precision types (`F64DenseMatrix`).

Choose a factorization that matches the system:

```kotlin
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.*

val system = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
val rhs = doubleArrayOf(3.0, 5.0)
val viaLu = system.lu().solve(rhs)
val viaCholesky = system.cholesky().solve(rhs)

val design = DenseMatrix.of(
    arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0, 1.0)),
)
val leastSquares = design.qr().solve(doubleArrayOf(1.0, 2.0, 3.0))
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

## Control memory

Use BLAS-style overloads with caller-owned arrays when a routine should write into
an existing destination:

```kotlin
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.koblas

val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
val x = doubleArrayOf(3.0, 5.0)
val y = DoubleArray(2)
koblas.gemv(alpha = 1.0, a = a, x = x, beta = 0.0, y = y)
```

For operations that need temporary storage, reuse a `Workspace` and pass an output
array to an `Into` operation. Reserve its buffers before a hot loop when allocations
must be avoided:

```kotlin
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas

val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
val factor = koblas.factor(a)
val rhs = doubleArrayOf(3.0, 5.0)
val solution = DoubleArray(2)
val workspace = Workspace().apply { reserve(size = 2, count = 1) }

repeat(1_000) { koblas.solveInto(factor, rhs, solution, workspace = workspace) }
```

## Backends

Every top-level operation uses Koblas’s process-wide backend registry. It starts with
portable implementations and selects the strongest registered provider for each dense
or sparse backend half.

On JVM, call `discoverBackends()` once at startup to probe host libraries, bundled
providers, and service-loaded providers. Available providers register automatically;
Koblas otherwise keeps using its portable implementation.

Use `registerBackend(...)` to add your own provider; explicit registrations outrank
discovered ones. Use `installBackends(...)` to replace the entire global context, and
`installBackends(null)` to restore registry selection. These JVM properties or
environment variables steer discovery to custom library paths:

| Library | JVM property | Environment variable |
|---------|--------------|----------------------|
| CBLAS | `koblas.cblas.path` | `KOBLAS_CBLAS_PATH` |
| LAPACKE | `koblas.lapacke.path` | `KOBLAS_LAPACKE_PATH` |
| SuiteSparse KLU 2 | `koblas.klu.path` | `KOBLAS_KLU_PATH` |
| UMFPACK | `koblas.umfpack.path` | `KOBLAS_UMFPACK_PATH` |
| BASICLU | `koblas.basiclu.path` | `KOBLAS_BASICLU_PATH` |

The JVM property takes precedence. The native targets read the environment variables only,
having no system properties, and only for the bindings they carry: CBLAS, LAPACKE and
UMFPACK. KLU and BASICLU are bound on the JVM alone, so their variables do nothing
elsewhere. Inspect `koblasInfo` or `koblas.portableSlots`
after configuration, or call `koblas.requireAccelerated(...)` to require acceleration.

These pin selection per half, the JVM property again first:

| Half | JVM property | Environment variable |
|------|--------------|----------------------|
| Dense | `koblas.dense.backend` | `KOBLAS_DENSE_BACKEND` |
| Sparse | `koblas.sparse.backend` | `KOBLAS_SPARSE_BACKEND` |

The value is a backend name, with the `-bundled` suffix optional and `reference` selecting
none of the host bindings. A blank setting counts as unset, leaving the half to priority.

## Multithreading

The portable reference and SIMD implementations are single-threaded. The JVM OpenBLAS
binding can use a threaded OpenBLAS build for supported dense operations, but automatic
discovery configures one thread. Configure and register it during startup instead:

```kotlin
import com.eignex.koblas.registerBackend
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.jvm.F64Backends

registerBackend(F64Backends(HostBlasConfig(threadCount = 8)))
```

Start the JVM with `-Xss16m` as well: threaded LAPACK needs the larger Java thread
stack and can otherwise crash the process. OpenBLAS owns this setting process-wide. The
bundled OpenBLAS is built with threading disabled, so bundled UMFPACK is also
single-threaded. KLU has no multithreaded mode; a host UMFPACK may use threads only
through the BLAS library it was built against.
