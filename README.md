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
| koblas-suitesparse | UMFPACK for general sparse LU, KLU for a fixed sparsity pattern. |
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
host libraries. `koblas-suitesparse` bundles OpenBLAS for dense operations; BASICLU does
not need it. Bundled modules have licenses in addition to Apache 2.0; see their generated
`THIRD-PARTY-NOTICES.txt` files for details. `koblas-suitesparse` carries GPL-licensed
SuiteSparse packages, which makes that artifact effectively GPL-3.0.

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
val x = a.lu().use { factor -> factor.solve(doubleArrayOf(4.0, 9.0)) }
```

Sparse factorizations are `AutoCloseable`. Use `use` for a bounded solve or retain the factor across solves
and call `close()` when finished. Native KLU, UMFPACK, CHOLMOD, BASICLU, and HFactor factors release their
resources immediately; cleaners remain only as protection for a factor the caller forgot to close. Closing
is idempotent. The dimension and singular status remain readable afterwards, while solves and native factor
statistics throw `IllegalStateException`.

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
portable implementations and selects the strongest registered provider for each semantic
role. Sparse general LU, repeated-pattern LU, Cholesky, LDL, basis factorization, and
basis solving are resolved independently. UMFPACK is the stable accelerated default for
ordinary sparse LU; registering KLU or BASICLU does not displace it from that role.

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

The JVM property takes precedence. Native targets have no system properties and read the
environment variables for the bindings they carry. Inspect `koblas.status` after
configuration, or call
`koblas.requireAccelerated(BackendRole.DENSE_BLAS)` to require a specific role:

```kotlin
import com.eignex.koblas.BackendExecution
import com.eignex.koblas.BackendRole
import com.eignex.koblas.F64RouteQuery
import com.eignex.koblas.koblas
import com.eignex.koblas.route
import com.eignex.koblas.status

val dense = koblas.status[BackendRole.DENSE_BLAS]
check(dense.available)

val route = koblas.route(F64RouteQuery.DenseGemm(m = 256, n = 64, k = 128))
check(route.execution == BackendExecution.NATIVE) {
    "${route.executor}: ${route.reason}, gate=${route.gate}"
}
```

Status describes the provider selected for each semantic role. A route additionally reports the predicted
executor, measured threshold, and fallback reason for a representative operation and problem shape. A
third-party backend that does not implement routing diagnostics reports `UNKNOWN`; this is distinct from a
known portable fallback or an unavailable binding. The legacy `koblasInfo` and slot-based inspection APIs
remain available.

Use typed capabilities for specialized sparse workflows. `backendNamed` retrieves an exact discovered
provider without narrowing to its implementation class; a role-specific context selection makes the
algorithm choice local to one solver:

```kotlin
import com.eignex.koblas.*

discoverBackends()
val klu = checkNotNull(backendNamed("klu", F64Capabilities.repeatedSparseLu))
val repeatedContext = F64ContextBuilder()
    .withBackend(BackendRole.SPARSE_REPEATED_LU, klu)
    .resolve()

val repeated = checkNotNull(repeatedContext.capability(F64Capabilities.repeatedSparseLu))
val initial = repeated.factor(a)
val updated = repeated.refactor(initial, samePatternWithNewValues)
```

Use `F64Capabilities.generalSparseLu` for unrelated systems,
`F64Capabilities.basisFactorizations` for BASICLU-style column replacement, and
`F64Capabilities.basisSolvers` for stateful simplex workflows. A specialized provider can still be selected
deliberately for general LU with `withBackend(BackendRole.SPARSE_GENERAL_LU, provider)`; only automatic
registry selection excludes repeated-pattern and basis-specialized providers from that default.

For solver-owned configuration, build and retain an explicit context. The builder is immutable: each
`with` call returns a new configuration, and `resolve()` neither reads nor mutates the global registry.

```kotlin
import com.eignex.koblas.BackendRole
import com.eignex.koblas.F64ContextBuilder
import com.eignex.koblas.F64DispatchPolicy

val strictDense = F64ContextBuilder()
    .withBackend(BackendRole.DENSE_BLAS, selectedBlas)
    .withBackend(BackendRole.DENSE_DECOMPOSITIONS, selectedLapack)
    .withDispatchPolicy(F64DispatchPolicy.NATIVE_ONLY)
    .resolve()

val c = strictDense.gemm(a, b)
```

`NATIVE_ONLY` rejects a known threshold, argument, availability, or portable-provider fallback before the
backend is invoked. `PORTABLE_ONLY` discards external selections and resolves every role to the reference
path. In `AUTO`, choose `ALLOW`, `WARN`, or `THROW` with `withFallbackPolicy`; `WARN` also requires an
`onFallback` handler. Enforcement applies to the operation families represented by `F64RouteQuery`, and the
route plan is available separately through `context.plan(query)`. Other routines retain their existing
backend behavior until their route families are added. Context policies never affect free functions, which
continue to use the process-wide `koblas` context.

These constrain discovery to one named provider, the JVM property again first. A sparse provider fills only
its semantic roles: pinning KLU selects repeated-pattern LU and its symmetric capabilities, while pinning
BASICLU selects basis factorization rather than changing general LU.

| Half | JVM property | Environment variable |
|------|--------------|----------------------|
| Dense | `koblas.dense.backend` | `KOBLAS_DENSE_BACKEND` |
| Sparse | `koblas.sparse.backend` | `KOBLAS_SPARSE_BACKEND` |

The value is a backend name, with the `-bundled` suffix optional and `reference` selecting
none of the host bindings. A blank setting counts as unset, leaving the half to priority.

## Native numerical options

Host-library paths belong to the `*Config` types; numerical and dispatch policy belongs to reusable
`*Options` values. The bundled and host-backed forms accept the same option type, so changing deployment
does not change solver policy:

```kotlin
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluOptions
import com.eignex.koblas.sparse.host.klu.KluOrdering
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import com.eignex.koblas.suitesparse.BundledKlu

val options = KluOptions(
    factorizeMin = 128,
    pivotTolerance = 0.01,
    useBtf = true,
    ordering = KluOrdering.AMD,
)
val systemKlu = KluSparseLu(KluConfig("/opt/suitesparse/lib/libklu.so.2", options))
val bundledKlu = BundledKlu(options)
```

The same split is available through `OpenBlasOptions`, `UmfpackOptions`, `BasicluOptions`, and
`HfactorOptions`. Existing configuration and bundled-provider constructors remain available. A provider's
`backendMetadata.options` reports the effective numerical settings and resolved dispatch gates; a requested
OpenBLAS thread count is reported only when the loaded library exposes and accepts the thread setter.

## Multithreading

The portable reference and SIMD implementations are single-threaded. The JVM OpenBLAS
binding can use a threaded OpenBLAS build for supported dense operations, but automatic
discovery configures one thread. Configure and register it during startup instead:

```kotlin
import com.eignex.koblas.registerBackend
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.cblas.OpenBlasOptions
import com.eignex.koblas.dense.host.jvm.F64Backends

registerBackend(F64Backends(HostBlasConfig(OpenBlasOptions(threadCount = 8))))
```

Start the JVM with `-Xss16m` as well: threaded LAPACK needs the larger Java thread
stack and can otherwise crash the process. OpenBLAS owns this setting process-wide. The
bundled OpenBLAS is built with threading disabled, so bundled UMFPACK is also
single-threaded. KLU has no multithreaded mode; a host UMFPACK may use threads only
through the BLAS library it was built against.
