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

Dense and sparse double-precision linear algebra for JVM and Kotlin/Native compute hosts. Koblas provides
dense and sparse BLAS operations, built-in C/SIMD kernels, and optional HFactor acceleration.

Koblas is a low-level building block for numerical and optimization software that owns its data and algorithms.
It exposes storage, allocation, workspace, implementation, and lifecycle decisions instead of hiding them behind a
data-frame or expression layer.

## Platforms and modules

The core is published for JVM, Linux x64/arm64 Kotlin/Native, and macOS arm64 Kotlin/Native. JavaScript, Wasm,
Windows Native, and Apple mobile targets are not published.

| Module | Published targets | Purpose |
|--------|-------------------|---------|
| koblas | JVM, Linux x64/arm64, macOS arm64 | Dense BLAS and sparse linear algebra with built-in C/SIMD kernels and portable references. |
| koblas-hfactor | JVM | Bundled HFactor for hypersparse simplex workflows. |

On JVM, add `--add-modules=jdk.incubator.vector` to enable the built-in SIMD kernels. Without the Vector API,
koblas uses its bundled C kernels when available and otherwise retains the same semantics through its scalar
implementation. HFactor is an optional JVM API and is loaded only when constructed and used directly.

The non-published `koblas-bench` module owns development-only OpenBLAS and oneMKL comparators. They are never
dependencies or resources of a published module. See [`koblas-bench/README.md`](koblas-bench/README.md) for
installation and runtime requirements. Bundled modules carry their own third-party notices.

## Quick start

Dense containers use column-major storage. Operators cover ordinary arithmetic and BLAS-backed products:

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

Sparse matrices are validated CSC with ascending row indices in each column. Construct them from columns or
coordinate triplets, then use sparse products, transpose, preparation, and triangular operations.

## Data and storage

Koblas currently implements one numerical family: `Double`. Its public types use concise names such as
`DenseMatrix` and `SparseMatrix`; a future element type would add its own distinct public family.

| Family | Scalar | Dense storage | Sparse storage | Sparse index |
|--------|--------|---------------|----------------|--------------|
| Double | Kotlin Double | DenseVector, DenseMatrix, strided views | SparseVector, CSC SparseMatrix | Kotlin Int |

Compatible DoubleArray and CSC buffers can be wrapped without copying. Dense matrices are column-major, and
native sparse bindings use 32-bit-index entry points so sparse indices do not need widening copies.

Owned dense containers can expose live borrowed panels, columns, and strided rows. A view retains its physical
offset, leading dimension, and increment:

```kotlin
import com.eignex.koblas.*

val storage = DenseMatrix.zero(512, 32)
val panel = storage.view(row = 64, rows = 128, column = 4, cols = 8)
val weights = DenseMatrix.zero(8, 2)
val output = DenseMatrix.zero(128, 2)

koblas.blas.gemm(
    alpha = 1.0,
    a = panel,
    transposeA = false,
    b = weights.asView(),
    transposeB = false,
    beta = 0.0,
    c = output.asView(),
)
```

Views are not serializable because they do not own their buffers. Strided BLAS preserves offsets, increments,
and leading dimensions through the built-in implementation without packing. Disjoint views may share a backing buffer, but
a strided destination must not overlap an input.

## Numerical routine coverage

Koblas deliberately exposes the following double-precision subset. The routine names identify the corresponding
BLAS or Sparse BLAS operation where one exists. Routines not listed here are not part of the supported numerical
subset.

| Family | Koblas operations | Standard routines |
|--------|-------------------|-------------------|
| BLAS level 1 | `dot`, `axpy`, `scale`, `norm2`, `asum`, `iamax`, `copy`, `swap`, `rotg`, `rot`, `rotmg`, `rotm` | `ddot`, `daxpy`, `dscal`, `dnrm2`, `dasum`, `idamax`, `dcopy`, `dswap`, `drotg`, `drot`, `drotmg`, `drotm` |
| BLAS level 2 | `gemv`, `symv`, `ger`, `syr`, `syr2`, `trsv`, `trmv` | `dgemv`, `dsymv`, `dger`, `dsyr`, `dsyr2`, `dtrsv`, `dtrmv` |
| BLAS level 3 | `gemm`, `symm`, `syrk`, `syr2k`, `trsm`, `trmm` | `dgemm`, `dsymm`, `dsyrk`, `dsyr2k`, `dtrsm`, `dtrmm` |
| Dense utility | `transpose`, `norm1`, `normInf`, `normFro`, row/column scaling | No direct BLAS routine |
| Sparse BLAS | CSC `gemv`, triangular `trsv`/`trsm` and `trmv`/`trmm`, sparse–dense `gemm`, sparse–sparse product, `transpose`, prepared repeated products | Sparse BLAS `usmv`, `ussv`, `ussm`, `usmm`; triangular multiply is `usmv`/`usmm` over a triangle, and product and preparation are Koblas operations |

This table documents the subset, not a roadmap. In particular, it does not imply support for the other routines in
the BLAS or Sparse BLAS specifications.

## Allocation and workspaces

BLAS-style and Into-suffixed overloads write into caller-owned destinations. A Workspace grows automatically
and retains temporary storage for reuse:

```kotlin
import com.eignex.koblas.*

val a = DenseMatrix.zero(64, 32)
val x = DoubleArray(a.rows)
val out = DoubleArray(a.cols)
val workspace = Workspace()

repeat(1_000) {
    koblas.gemv(1.0, a, x, 0.0, out, transpose = true, workspace = workspace)
}
```

BLAS options use named Boolean parameters such as lower, transpose, unitDiag, and right.

## Built-in implementation

Top-level functions use the immutable `koblas` engine selected once for the platform. On JVM it prefers the
Vector API, then Koblas's bundled C kernels, then exact scalar kernels. Kotlin/Native uses the bundled C kernels,
with scalar fallbacks where needed. Shared dense and sparse matrix algorithms are bound to the engine's selected
kernels; there is no provider registry, service discovery, or process-global override.

Tests and benchmarks can construct an independent exact engine without changing global state:

```kotlin
@OptIn(ExperimentalKoblasApi::class)
val scalar = BuiltinKernels.scalar.engine()
val c = scalar.gemm(a, b)
```

`koblasInfo`, `KoblasContext.name`, and the kernel names provide read-only attribution for logs.

### Implementation configuration

On JVM, a system property takes precedence over the corresponding environment variable. Kotlin/Native reads
the environment variable. The JVM-only `koblas.jvm.vector.scatter` setting (or
`KOBLAS_JVM_VECTOR_SCATTER`) selects indexed Vector API stores for sparse kernels: auto (the default) makes a
conservative guess from a 512-bit x86 preferred species. Use on when you know the deployment has a profitable
AVX-512 path; it forces indexed stores when the Vector API module is present. Off retains scalar indexed
stores.

## Sparse workflows

Portable sparse Cholesky, LDL, LU, QR, symbolic analysis, and basis factorization have been removed. The
optional JVM `koblas-hfactor` artifact supplies general sparse LU and the stateful basis solver API directly:

```kotlin
import com.eignex.koblas.hfactor.BundledHfactor
import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu

val bundled = BundledHfactor()
check(bundled.availability.available) { bundled.availability.reason }
bundled.factor(a).use { factors -> factors.solveInto(rhs, solution) }

val explicit = HfactorSparseLu(HfactorConfig(libraryPath = "/opt/lib/libkoblas_hfactor.so.1"))
```

Existing direct `HfactorSparseLu` and `BundledHfactor` construction keeps the same imports. Code that selected
HFactor through `ContextBuilder`, `Capabilities`, or registry discovery must instead hold one of these objects
and call `factor` or `basisSolver` on it.

Repeated sparse products can retain an immutable CSC snapshot so a native backend marshals its descriptor once:

```kotlin
import com.eignex.koblas.sparse.prepare

a.prepare().use { prepared ->
    repeat(iterations) {
        prepared.gemv(1.0, x, 0.0, y)
    }
}
```

Prepared handles are AutoCloseable. HFactor factors and basis solvers own native resources and must also be
closed deterministically.

## Native options and threading

An explicit HFactor library path belongs to `HfactorConfig`; bundled and explicit-path construction accept the
same `HfactorOptions`, so the loading choice does not change numerical policy.

The portable reference, JVM SIMD, bundled C kernels, and HFactor are single-threaded.

## Ownership and concurrency

| Object | Contract |
|--------|----------|
| KoblasContext | Immutable after construction and safe to share. |
| Owned dense and sparse containers | Mutable and unsynchronized; concurrent reads require no reachable writer. |
| Strided views | Borrow live storage; the owner must outlive every use. |
| Workspace | Caller-owned scratch; use one per concurrent operation or serialize access. |
| HFactor factors and basis solvers | Caller-owned AutoCloseable resources; do not race use or update with close. |
| Prepared sparse descriptors | Immutable snapshots with externally serialized native workspace; close explicitly. |
| Destination-passing operations | Follow the documented alias contract for that operation. |

Native cleaners are leak guards only. Deterministic close or use remains the lifecycle contract, and a
strict allocation policy covers an operation call rather than factor construction, library initialization, or
thread-pool startup.
