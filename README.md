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
BLAS/LAPACK operations, factorizations, and optional OpenBLAS, SuiteSparse, BASICLU, or HFactor acceleration.

Koblas is a low-level building block for numerical and optimization software that owns its data and algorithms.
It exposes storage, allocation, workspace, backend, and lifecycle decisions instead of hiding them behind a
data-frame or expression layer.

## Platforms and modules

The core is published for JVM, Linux x64/arm64 Kotlin/Native, and macOS arm64 Kotlin/Native. JavaScript, Wasm,
Windows Native, and Apple mobile targets are not published.

| Module | Published targets | Purpose |
|--------|-------------------|---------|
| koblas | JVM, Linux x64/arm64, macOS arm64 | Dense and sparse API with a portable reference backend. |
| koblas-openblas | JVM | Bundled OpenBLAS and LAPACKE. |
| koblas-suitesparse | JVM | Bundled SuiteSparse and its OpenBLAS dependency. |
| koblas-basiclu | JVM | Bundled BASICLU for simplex-basis factorization. |
| koblas-hfactor | JVM | Bundled HFactor for hypersparse simplex workflows. |

On JVM, add `--add-modules=jdk.incubator.vector` to enable the built-in SIMD kernels. Host OpenBLAS and
SuiteSparse libraries are discovered when installed; the bundled modules take precedence on JVM Linux
x64/arm64 and macOS arm64. When an accelerated provider is unavailable, koblas retains the same semantics
through its portable implementation.

Bundled modules carry their own third-party notices. In particular, koblas-suitesparse includes GPL-licensed
SuiteSparse packages and is therefore effectively GPL-3.0.

## Quick start

Dense containers use column-major storage. Operators cover ordinary arithmetic and products, while typed
factorizations expose solve operations:

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
val solution = a.cholesky().solve(x.data)
```

Sparse matrices are validated CSC with ascending row indices in each column. Construct them from columns or
coordinate triplets and close native-backed factors deterministically:

```kotlin
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.lu

val a = SparseMatrix.ofColumns(
    rows = 2,
    cols = 2,
    columns = listOf(listOf(0 to 2.0), listOf(1 to 3.0)),
)
val solution = a.lu().use { factor ->
    factor.solve(doubleArrayOf(4.0, 9.0))
}
```

## Data and storage

Koblas currently implements one numerical family. The element type is explicit in expert-facing public names,
while unqualified aliases such as `DenseMatrix` and `SparseMatrix` name the F64 types.

| Family | Scalar | Dense storage | Sparse storage | Sparse index |
|--------|--------|---------------|----------------|--------------|
| F64 | Kotlin Double | F64DenseVector, F64DenseMatrix, strided views | F64SparseVector, CSC F64SparseMatrix | Kotlin Int |

Compatible DoubleArray and CSC buffers can be wrapped without copying. Dense matrices are column-major, and
native SuiteSparse bindings use 32-bit-index entry points so sparse indices do not need widening copies.

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

Views are not serializable because they do not own their buffers. Strided BLAS passes offsets, increments, and
leading dimensions to a selected host backend without packing. Disjoint views may share a backing buffer, but
a strided destination must not overlap an input.

## Numerical routine coverage

Koblas deliberately exposes the following double-precision subset. The routine names identify the corresponding
BLAS, LAPACK, or Sparse BLAS operation where one exists; the portable backend defines the same semantics when a
host provider does not. Routines not listed here are not part of the supported numerical subset.

| Family | Koblas operations | Standard routines |
|--------|-------------------|-------------------|
| BLAS level 1 | `dot`, `axpy`, `scale`, `asum`, `iamax`, `copy`, `swap`, `rotg`, `rot` | `ddot`, `daxpy`, `dscal`, `dasum`, `idamax`, `dcopy`, `dswap`, `drotg`, `drot` |
| BLAS level 2 | `gemv`, `symv`, `ger`, `syr`, `syr2`, `trsv`, `trmv` | `dgemv`, `dsymv`, `dger`, `dsyr`, `dsyr2`, `dtrsv`, `dtrmv` |
| BLAS level 3 | `gemm`, `symm`, `syrk`, `syr2k`, `trsm`, `trmm` | `dgemm`, `dsymm`, `dsyrk`, `dsyr2k`, `dtrsm`, `dtrmm` |
| Dense utility | `transpose`, `norm1`, `normInf`, `normFro`, row/column scaling | No direct BLAS routine |
| Dense LAPACK | LU (`factor`, `solve`, `invert`, `rcond`), pivoted LDL, QR and pivoted QR, Cholesky, triangular inverse | `dgetrf`, `dgetrs`, `dgetri`, `dgecon`, `dsytrf`, `dsytrs`, `dgeqrf`, `dgeqp3`, `dormqr`, `dpotrf`, `dpotrs`, `dpotri`, `dtrtri` |
| Sparse BLAS | CSC `gemv`, triangular `trsv`/`trsm`, sparse–dense `gemm`, sparse–sparse product, `transpose`, prepared repeated products | Sparse BLAS `usmv`, `ussv`, `ussm`, `usmm`; product and preparation are Koblas operations |
| Sparse factorizations | General LU, repeated-pattern LU, Cholesky, LDL, QR, and simplex basis operations | Provider-specific SuiteSparse, BASICLU, and HFactor capabilities |

This table documents the subset, not a roadmap. In particular, it does not imply support for the other routines in
the BLAS, LAPACK, or Sparse BLAS specifications.

## Allocation and workspaces

BLAS-style and Into-suffixed overloads write into caller-owned destinations. A Workspace grows automatically
and retains temporary storage for reuse:

```kotlin
import com.eignex.koblas.*

val factor = koblas.factor(a)
val rhs = doubleArrayOf(3.0, 5.0)
val out = DoubleArray(2)
val workspace = Workspace()

repeat(1_000) {
    koblas.solveInto(factor, rhs, out, workspace = workspace)
}
```

Sparse factors report their exact scratch requirement through `solveAllocation(aliasing, transpose)`. Reserve
those buffers and select an AllocationPolicy when the guarantee should be enforced before destination
mutation. REQUIRE_NO_SIZE_DEPENDENT_MANAGED permits fixed JVM/FFM overhead; stronger policies are reported
only where koblas controls the corresponding allocation source.

Dense decompositions hold only Kotlin arrays, so they have no `close()`, native resource lifecycle, symbolic
analysis, or prepared-handle API. Reuse a dense factor for repeated right-hand sides; for same-sized changing
matrices, refactor it in place with `factor.refactorInto(nextMatrix)`. The dense `solveInto` extensions retain
the destination you pass and accept a Workspace whenever their backend needs staging, but do not advertise a
cross-backend allocation guarantee: a selected native LAPACK provider may own additional temporary storage.

BLAS options use named Boolean parameters such as lower, transpose, unitDiag, and right.

## Backends and routing

Every operation runs through an F64Context. Top-level functions use the process-wide koblas context, whose
registry selects providers independently by semantic role. General sparse LU, repeated-pattern LU, Cholesky,
quasi-definite LDL, QR, basis factorization, and basis solving are separate choices rather than one
interchangeable sparse backend.

Selected providers execute their native implementations at every size. They fall back only for unavailable
libraries, unsupported arguments, or operations they do not implement. Inspect status for the selected providers
and route a representative problem before entering a hot loop:

```kotlin
import com.eignex.koblas.*

discoverBackends()
val dense = koblas.status[BackendRole.DENSE_BLAS]
check(dense.available)

val route = koblas.route(F64RouteQuery.DenseGemm(m = 256, n = 64, k = 128))
check(route.execution == BackendExecution.NATIVE) {
    "${route.executor}: ${route.reason}"
}
```

The `registerBackend(...)` function adds a provider explicitly. The `installBackends(...)` function replaces the
process-wide context; passing null restores registry selection. For solver-local control, use an immutable
F64ContextBuilder:

```kotlin
val strictDense = F64ContextBuilder()
    .withBackend(BackendRole.DENSE_BLAS, selectedBlas)
    .withBackend(BackendRole.DENSE_DECOMPOSITIONS, selectedLapack)
    .withDispatchPolicy(F64DispatchPolicy.NATIVE_ONLY)
    .resolve()

val c = strictDense.gemm(a, b)
```

NATIVE_ONLY rejects a known fallback before backend invocation. PORTABLE_ONLY resolves every role to the
reference implementation. In AUTO, fallback can be allowed, reported to a handler, or rejected. Third-party
providers without route diagnostics report UNKNOWN rather than being assumed native.

### Discovery configuration

On JVM, a system property takes precedence over the corresponding environment variable. Kotlin/Native reads
the environment variable. Override a library path with the JVM property `koblas.<library>.path` or environment
variable `KOBLAS_<LIBRARY>_PATH`. Supported library identifiers are cblas, lapacke, klu, umfpack, cholmod,
basiclu, and hfactor. The JVM-only `koblas.jvm.vector.scatter` setting (or
`KOBLAS_JVM_VECTOR_SCATTER`) selects indexed Vector API stores for sparse kernels: auto (the default) makes a
conservative guess from a 512-bit x86 preferred species. Use on when you know the deployment has a profitable
AVX-512 path; it forces indexed stores when the Vector API module is present. Off retains scalar indexed
stores.

Pin discovery by backend name per semantic role. Set a JVM property named `koblas.backend.<role>` or the
matching `KOBLAS_<ROLE>_BACKEND` environment variable; the property takes precedence. A blank value leaves the
role automatic, while `reference` disables host selection for that role. For example, pin general sparse LU
with `koblas.backend.sparse.general.lu` or `KOBLAS_SPARSE_GENERAL_LU_BACKEND`.

## Sparse workflows

Choose a semantic capability based on the matrix sequence and numerical structure:

| Workload | Capability | Typical provider | Constraint |
|----------|------------|------------------|------------|
| Unrelated general systems | generalSparseLu | UMFPACK | Numerical pivoting; stable ordinary-LU role. |
| Same CSC pattern, changing values | repeatedSparseLu | KLU | Analyze once; ordered CSC pattern must match exactly. |
| Symmetric positive-definite systems | sparseCholesky | CHOLMOD | Reads the lower triangle and rejects a non-positive pivot. |
| Quasi-definite KKT systems | quasiDefiniteLdl | CHOLMOD | Numerically unpivoted; use general LU for arbitrary indefinite matrices. |
| Overdetermined least-squares systems | sparseQr | SPQR | Requires at least as many rows as columns. |
| Simplex basis column replacement | basisFactorizations | BASICLU | Each update supersedes the preceding factor. |
| Stateful simplex solve/update loop | basisSolvers | HFactor | Own and close the solver; use typed ftran, btran, and update. |

Each sparse factorization returns the factor type its own kind names, and each exposes its factors: an LU
carries L, U, the two orderings and the row scaling; a Cholesky and quasi-definite LDL carry L, their ordering
and, for quasi-definite LDL, D; a QR carries R, the column ordering, the estimated rank and Q as an operator
through applyQInto.
Sparse QR is the one whose factor is not an F64SparseFactorization, because an m-by-n factorization takes a
right-hand side of length m and answers one of length n.

Factors materialise on first read and cost a copy out of the library, so solving alone never pays for them. A
provider that keeps its factors in a form it cannot return raises FactorsNotExposed.

Use the typed capability selected in the context rather than casting to a provider implementation.
Repeated-pattern LU, for example, retains symbolic analysis across numeric factors:

```kotlin
import com.eignex.koblas.*

discoverBackends()
val repeated = koblas.repeatedSparseLu
    ?: error("Repeated sparse LU is unavailable")

repeated.analyze(a).use { analysis ->
    analysis.factor(a).use { initial ->
        analysis.refactor(initial, samePatternWithNewValues).use { updated ->
            updated.solveInto(rhs, out)
        }
    }
}
```

Repeated sparse products can retain an immutable CSC snapshot so a native backend marshals its descriptor once:

```kotlin
import com.eignex.koblas.sparse.prepare

a.prepare().use { prepared ->
    repeat(iterations) {
        prepared.gemv(1.0, x, 0.0, y)
    }
}
```

Prepared handles and sparse factors are AutoCloseable. Native block solves accept a column-major dense matrix
of right-hand sides. Sparse quasi-definite LDL factors expose their pivot-sign inertia directly. Dense
`pivotedSymmetricIndefinite` is Bunch-Kaufman numerically pivoted for stability; it is not interchangeable
with sparse `quasiDefiniteLdl`, whose ordering controls fill.

## Factorization coverage

The implemented factor families deliberately have different capabilities: matrix shape, numerical meaning,
and ownership determine what is useful rather than forcing every factor into one interface.

| Family | Solve / transpose / blocks | Reuse and lifecycle | Safe factor inspection | Deliberate non-applicability |
|--------|----------------------------|---------------------|------------------------|------------------------------|
| Dense LU | Vector and column-major multi-RHS `solve`/`solveInto`, including transpose | `factorInto` reuses packed buffers; Kotlin-owned factors do not close | `lowerFactor`, `upperFactor`, `rowOrder`, singularity, determinant/sign/log-absolute determinant, inverse and `rcond` | — |
| Dense Cholesky | Vector and multi-RHS `solve`/`solveInto`; transpose is identical by symmetry | Pure Kotlin buffers; no symbolic lifecycle | `lowerFactor`, lower packed factor, inverse | No separate transpose solve or symbolic analysis |
| Dense LDL | Vector and multi-RHS `solve`/`solveInto`; transpose is identical by symmetry | Pure Kotlin buffers; no symbolic lifecycle | `packedFactor`, `pivotBlocks`, singularity | Bunch-Kaufman packing has no independent unpermuted `L`/diagonal `D` snapshot |
| Dense QR / pivoted QR | Q application, least-squares solve and `solveInto`; pivoted QR reports rank | Workspace reuses solve/Q scratch; pure Kotlin buffers do not close | `explicitQ`, `explicitR`, pivoted `columnOrder`, rank | No inverse or square-system transpose solve for rectangular QR |
| Triangular inversion | `trsv`/`trsm` supply vector and multi-RHS normal/transpose solves | Stateless; no factor or lifecycle | `trtri` returns the selected inverse triangle | No retained factorization or symbolic phase |
| Sparse LU | Vector and multi-RHS `solve`/`solveInto`, including transpose and alias-safe defaults | Native factors close; repeated-pattern LU has `analyze`/`refactor` | L/U, orderings, scaling, off-diagonal, fill, pivot quality, singularity where providers can expose them | No general sparse inverse or determinant API |
| Sparse Cholesky / LDL | Vector and multi-RHS factor solves; transpose is identical by symmetry | Native factors close; portable factors close as no-ops | L/order; LDL D/inertia | No separate transpose solve or dense inverse |
| Sparse QR | `applyQ`/`applyQInto`, vector and multi-RHS least-squares solve | Native factor lifecycle; workspace block staging | R, column order, rank and fill | Q remains an operator; explicit Q/inverse is generally dense and is intentionally not materialized |
| Basis factorizations / solvers | Basis factors inherit LU solves; solvers provide FTRAN/BTRAN | Column replacement, refactorization, update count, and close where native-owned | Basis factor exposes normal LU inspection; solver reports dimension/fill/updates/singularity | No matrix inverse, determinant, or generic multi-RHS API for hypersparse indexed-vector workflows |

## Native options and threading

Library paths belong to provider configuration types, while numerical and dispatch policy belongs to reusable
options values. Bundled and host-backed providers accept the same options, so deployment can change without
changing numerical policy. Effective options and resolved gates appear in `backendMetadata.options`.

The portable reference and JVM SIMD implementations are single-threaded. Automatic OpenBLAS discovery selects
one thread; configure and register a host backend explicitly to request more:

```kotlin
import com.eignex.koblas.registerBackend
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.cblas.OpenBlasOptions
import com.eignex.koblas.dense.host.jvm.F64Backends

registerBackend(F64Backends(HostBlasConfig(OpenBlasOptions(threadCount = 8))))
```

Threaded LAPACK on JVM also needs `-Xss16m`; an insufficient Java thread stack can crash the process. OpenBLAS
thread configuration is process-wide. Bundled OpenBLAS is built without threading, KLU is single-threaded, and
a host UMFPACK can use threads only through the BLAS against which it was built.

## Ownership and concurrency

| Object | Contract |
|--------|----------|
| F64Context, status, and route values | Immutable after resolution and safe to share. |
| Global backend registry | Configure during startup; process-wide selection is intentionally global. |
| Owned dense and sparse containers | Mutable and unsynchronized; concurrent reads require no reachable writer. |
| Strided views | Borrow live storage; the owner must outlive every use. |
| Workspace | Caller-owned scratch; use one per concurrent operation or serialize access. |
| Sparse factors and symbolic analyses | Caller-owned AutoCloseable resources; do not race use or refactor with close. |
| Prepared sparse descriptors | Immutable snapshots with externally serialized native workspace; close explicitly. |
| Destination-passing operations | Follow the documented alias contract for that operation. |

Native cleaners are leak guards only. Deterministic close or use remains the lifecycle contract, and a
strict allocation policy covers an operation call rather than factor construction, library initialization, or
thread-pool startup.
