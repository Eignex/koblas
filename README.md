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

## Element types

Koblas currently implements one numerical element family. The type is part of every public expert-facing
name so additional precisions can be added without changing the meaning or registry of the existing one.

| Family | Scalar | Dense storage | Sparse storage | Sparse index |
|--------|--------|---------------|----------------|--------------|
| `F64` | Kotlin `Double` | `F64DenseVector`, `F64DenseMatrix`, and strided borrowed views | `F64SparseVector` and validated CSC `F64SparseMatrix` | Kotlin `Int` (32-bit) |

Unqualified aliases such as `DenseMatrix`, `SparseMatrix`, and `DenseVector` name the `F64` types. Native
SuiteSparse bindings deliberately use their 32-bit-index entry points so CSC indices do not require widening
copies. BF16 is a planned family, not a currently selectable precision or backend role.

## Use

Containers use column-major storage. Unqualified names such as `DenseMatrix`
are aliases for the implemented double-precision types (`F64DenseMatrix`).
Owned containers may expose live borrowed panels and slices without copying. A matrix view retains its
physical leading dimension, while a row is a strided vector view:

```kotlin
import com.eignex.koblas.*
import com.eignex.koblas.core.*
import com.eignex.koblas.dense.Transpose

val storage = DenseMatrix.zero(512, 32)
val panel = storage.view(row = 64, rows = 128, column = 4, cols = 8)
val row = panel.row(7)
val weights = DenseMatrix.zero(8, 2)
val output = DenseMatrix.zero(128, 2)
check(storage.ownership == BufferOwnership.OWNED)
check(panel.ownership == BufferOwnership.BORROWED)

koblas.blas.gemm(
    1.0,
    panel,
    Transpose.NO_TRANSPOSE,
    weights.asView(),
    Transpose.NO_TRANSPOSE,
    0.0,
    output.asView(),
)
```

Views are live and not serializable because they do not own their buffers. Strided `gemv` and `gemm` pass
offsets, increments, and leading dimensions directly to host BLAS where the measured gate selects it; their
destinations may share a buffer with disjoint views but must not overlap an input.

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
matrix-vector products, and triangular solves. Their factorizations include general LU, repeated-pattern LU,
Cholesky, and unpivoted `L·D·Lᵀ`, with separate basis-factorization and basis-solver roles for simplex updates.
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

Every sparse factor accepts a column-major `F64DenseMatrix` of right-hand sides and a caller-owned output.
The portable default is alias-safe and solves column by column; KLU and CHOLMOD pass the whole block through
their native ABIs in one foreign call. `factor.report()` returns common diagnostics with nullable unavailable
fields, plus provider-specific details. A valid statistic of zero therefore remains distinguishable from one
the provider cannot report.

Repeated products can retain an immutable CSC snapshot with `prepare()` so a native backend marshals its
sparse descriptor only once:

```kotlin
import com.eignex.koblas.sparse.prepare

a.prepare().use { prepared ->
    repeat(iterations) {
        prepared.gemv(1.0, x, 0.0, y)
    }
}
```

The source may be changed after preparation without affecting the snapshot. Prepared handles are
`AutoCloseable`; do not use or close one concurrently without external serialization. CHOLMOD exposes
separate prepared `gemv`, dense-product, and sparse-product gates because amortized crossovers differ from
one-shot calls; set the corresponding `CholmodSparseBlas` constructor gate to zero to force a measured path.

## Typed BLAS flags

Expert code can use `Uplo`, `Transpose`, `Diag`, and `Side` instead of Boolean flag clusters. The enums make
call sites self-describing and prevent argument-order mistakes. Existing Boolean overloads remain available
for source compatibility and delegate to the same backend seam.

```kotlin
import com.eignex.koblas.dense.*

trsm(
    a = triangle,
    b = rightHandSides,
    uplo = Uplo.LOWER,
    transpose = Transpose.NO_TRANSPOSE,
    diag = Diag.NON_UNIT,
    side = Side.LEFT,
)
```

For migration, `lower = true/false` maps to `Uplo.LOWER/UPPER`, `transpose = false/true` to
`Transpose.NO_TRANSPOSE/TRANSPOSE`, `unitDiag = false/true` to `Diag.NON_UNIT/UNIT`, and `right = false/true`
to `Side.LEFT/RIGHT`. `Uplo.FULL` remains valid for koblas operations that deliberately update both symmetric
triangles, such as `syr`; BLAS operations that must select exactly one stored triangle reject it before
mutating a destination.

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

`Workspace` pools `DoubleArray` and `IntArray` scratch independently. Sparse factors expose a structured
`solveAllocation(aliasing, transpose)` capability. Its scratch list is the exact managed storage that must
already be idle in the workspace for the reported guarantee to hold. A strict call rejects an unsupported
guarantee or missing reservation before touching the destination:

```kotlin
import com.eignex.koblas.AllocationPolicy
import com.eignex.koblas.Workspace

val solveWorkspace = Workspace()
sparseFactor.solveAllocation(aliasing = false).scratch.forEach(solveWorkspace::reserve)
repeat(1_000) {
    sparseFactor.solveInto(
        sparseRhs,
        sparseSolution,
        workspace = solveWorkspace,
        allocationPolicy = AllocationPolicy.REQUIRE_NO_SIZE_DEPENDENT_MANAGED,
    )
}
```

`NO_SIZE_DEPENDENT_MANAGED` permits fixed JVM/FFM call overhead. `NO_MANAGED` additionally rules that out but
does not claim that an external library avoids native allocation. `NO_MANAGED_OR_NATIVE` is therefore
reported only where koblas controls the complete hot path. Native factor objects and their retained scratch
remain externally serialized as described above.

## BLAS coverage

`P` means koblas has a portable implementation on every target. `N` means the named provider has a native
implementation, subject to availability, supported arguments, and its measured gate; below a gate it runs
the same portable semantics. This table describes reachable implementations, not the provider a process has
selected. Inspect `context.plan(query)` for the decision of one concrete shape.

| Routine family | Reference | OpenBLAS / CBLAS | CHOLMOD sparse BLAS |
|----------------|-----------|-------------------|----------------------|
| `dot`, `axpy`, `scale`, `copy`, `swap`, `norm2`, `asum`, `iamax` | P | N where selected; JVM normally prefers its SIMD kernels | — |
| Dense `gemv`, `ger`, `symv`, `trsv`, `trmv` | P | N above the level-2 gate | — |
| Dense `gemm`, `syrk`, `symm`, `trsm`, `trmm` | P | N above the level-3 gate | — |
| Dense `transpose`, `syr`, `syr2`, `syr2k` | P | P by measured or implementation decision | — |
| Strided dense `gemv`, `gemm` with positive strides | P | N with offset, increment, and leading dimension preserved | — |
| Sparse `gemv` | P | — | P automatically; native prepared entry point remains forceable for measurement |
| Sparse-dense `gemm` | P | — | N for a left sparse operand and untransposed dense operand above its one-shot or prepared gate |
| Sparse-sparse `gemm`, sparse `transpose`, `trsv`, `trsm` | P | — | P by measured or unsupported-operation decision |

| Factorization / solve family | Reference | LAPACKE | UMFPACK | KLU | CHOLMOD | BASICLU | HFactor |
|------------------------------|-----------|---------|---------|-----|---------|---------|---------|
| Dense LU factor, block solve, inverse, `rcond`, `trtri` | P | N above operation-specific gates | — | — | — | — | — |
| Dense Cholesky, LDL, QR, pivoted QR, `applyQ` | P | N above gates where the ABI is bound; QR solves and one-RHS LDL solve stay P | — | — | — | — | — |
| General sparse LU | P | — | N, stable automatic host default | N through its explicit repeated-pattern provider | — | N through its explicit basis provider | N as a lower-priority general provider |
| Same-pattern symbolic analysis and numeric refactor | — | — | — | N | — | — | — |
| Sparse Cholesky and unpivoted sparse LDL | P | — | — | — | N | — | — |
| Sparse vector solve | P | — | N | N | N | N | N through basis workflows |
| Sparse dense-block solve | P column-wise | — | N with one call per RHS | N in one `nrhs` call | N in one dense-block call | N with one call per RHS | N with one call per RHS |
| Basis column replacement | P by refactorization | — | — | — | — | N | N through the stateful basis-solver API |

Provider code is compiled for these targets:

| Provider | Targets |
|----------|---------|
| Portable reference | JVM, JS, Wasm JS, Wasm WASI, Linux, macOS, Windows, and iOS |
| JVM SIMD kernels | JVM with `jdk.incubator.vector` |
| Host OpenBLAS/LAPACKE and SuiteSparse | JVM plus Linux and macOS Kotlin/Native when the libraries resolve |
| Bundled OpenBLAS and SuiteSparse modules | JVM Linux x64/arm64 and JVM macOS arm64 |
| Host or bundled BASICLU | JVM; host bindings also compile for Linux and macOS Kotlin/Native |
| HFactor | JVM |

SPQR is intentionally absent: the produced SuiteSparse artifact no longer builds or packages an unreachable
SPQR library. Windows, iOS, JS, and Wasm therefore use the portable matrix and factorization implementations.

### Routing and fallback contracts

Native availability never implies native execution. The effective gates are provider options and appear in
`BackendMetadata.options`; route-aware families report their exact comparison through `DispatchGate`.

| Query family | Gate metric | Common portable reasons |
|--------------|-------------|-------------------------|
| `DenseGemv` | smaller matrix dimension | below threshold or unavailable CBLAS |
| `DenseGemm` | saturated `m·n·k` work against the level-3 crossover | below threshold |
| `DenseLu` | matrix order | below factorization threshold |
| `SparseLu` | stored CSC entries | below provider factorization threshold or unavailable provider |
| `SparseDenseGemm` | stored CSC entries | below threshold, right-side sparse operand, or transposed dense operand |
| `PreparedSparseProduct` | stored CSC entries with separate `GEMV`, `DENSE_GEMM`, and `SPARSE_GEMM` gates | measured portable decision or unavailable CHOLMOD |
| `SparseTriangularSolve` | stored CSC entries plus RHS count and flags | no bound provider currently implements the caller-matrix solve |

`NATIVE_ONLY` and fallback `THROW` reject a known fallback before destination mutation. `WARN` invokes the
configured handler first. A third-party provider without route diagnostics reports `UNKNOWN`; strict native
policy rejects that uncertainty rather than assuming acceleration. Operations without an `F64RouteQuery`
family retain backend behavior but cannot yet be preflighted by a context policy.

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
val solution = DoubleArray(a.rows)
repeated.analyze(a).use { analysis ->
    analysis.factor(a).use { initial ->
        analysis.refactor(initial, samePatternWithNewValues).use { updated ->
            updated.solveInto(rhs, solution)
        }
    }
}
```

The analysis snapshots the exact CSC structure and rejects a different pattern before native numeric work.
Numeric factors remain caller-owned and are closed inside the analysis lifetime. This keeps a repeated solve
typed at the capability boundary; no cast to KLU or another implementation is required.

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

## Choosing a sparse workflow

| Workload | Preferred semantic capability | Typical provider | Important constraint |
|----------|-------------------------------|------------------|----------------------|
| Unrelated general systems | `generalSparseLu` | UMFPACK, otherwise reference | Uses numerical pivoting; this is the stable ordinary-LU role |
| Same CSC pattern with changing values | `repeatedSparseLu` | KLU | Analyze once, require an exact ordered CSC-pattern match, and close numeric factors before the analysis |
| Symmetric positive-definite systems | `sparseCholesky` | CHOLMOD, otherwise reference | Reads the lower triangle and rejects the first non-positive pivot |
| Quasi-definite interior-point KKT systems | `sparseLdl` | CHOLMOD, otherwise reference | Sparse LDL is unpivoted numerically; use general LU for an arbitrary indefinite matrix |
| Simplex basis with column replacement | `basisFactorizations` | BASICLU | The replacement returns a superseding factor and closes the old native resource |
| Stateful simplex factor/solve/update loop | `basisSolvers` | HFactor | Own and close the solver; use `ftran`, `btran`, and `update` through the typed capability |

### Complete controlled solve

This example pins the semantic provider locally, proves that the concrete matrix shape will take a native
route, reserves the factor's declared scratch, enforces its allocation contract before mutation, samples the
report, and releases the factor deterministically:

```kotlin
import com.eignex.koblas.*

discoverBackends()
val umfpack = checkNotNull(backendNamed("umfpack", F64Capabilities.generalSparseLu))
val context = F64ContextBuilder()
    .withBackend(BackendRole.SPARSE_GENERAL_LU, umfpack)
    .withDispatchPolicy(F64DispatchPolicy.NATIVE_ONLY)
    .resolve()

val plan = context.plan(F64RouteQuery.SparseLu(a.nnz))
check(plan.decision == BackendPolicyDecision.EXECUTE) { plan.route }

val out = DoubleArray(a.rows)
val workspace = Workspace()
context.factor(a).use { factor ->
    val allocation = factor.solveAllocation(aliasing = false)
    allocation.scratch.forEach(workspace::reserve)
    factor.solveInto(
        b = rhs,
        out = out,
        workspace = workspace,
        allocationPolicy = AllocationPolicy.REQUIRE_NO_SIZE_DEPENDENT_MANAGED,
    )
    check(factor.report().provider == "umfpack")
}
```

For a repeated-pattern loop, select the KLU capability and retain its symbolic analysis. `refactor` checks
the full ordered CSC structure before native numeric work and supersedes the preceding factor:

```kotlin
val klu = checkNotNull(backendNamed("klu", F64Capabilities.repeatedSparseLu))
val repeatedContext = F64ContextBuilder()
    .withBackend(BackendRole.SPARSE_REPEATED_LU, klu)
    .resolve()
val repeated = checkNotNull(repeatedContext.capability(F64Capabilities.repeatedSparseLu))

repeated.analyze(a).use { analysis ->
    analysis.factor(a).use { initial ->
        analysis.refactor(initial, samePatternWithNewValues).use { updated ->
            updated.solveInto(rhs, out, workspace = workspace)
        }
    }
}
```

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

## Ownership, lifecycle, and concurrency

| Object or operation | Ownership and concurrency contract |
|---------------------|------------------------------------|
| `F64Context` and resolved status/route values | Immutable after resolution and safe to share. Construct separate contexts for different policies. |
| Global backend registry | Configure during startup. Registration is atomic, but changing the process-wide selection while work is running makes operation-to-operation choice intentionally global. |
| Owned dense/sparse containers | Mutable through their buffers and not synchronized. Concurrent reads are safe only while no writer can reach the storage. |
| Strided views | Borrow their backing and remain live. The owner must outlive every use. `gemv`/`gemm` reject output overlap with an input, while disjoint views may share one buffer. |
| `Workspace` | Caller-owned reusable scratch. Give each concurrent operation its own workspace or externally serialize access. Reserve before entering an allocation-sensitive loop. |
| Sparse factors and symbolic analyses | Caller-owned `AutoCloseable` resources. Close numeric factors before their analysis. Do not race solve/refactor/report with close; externally serialize a shared native factor. |
| Prepared sparse descriptors | Immutable snapshots of source CSC data, but their native common/workspace is externally serialized. Close explicitly and do not race use with close. |
| Destination-passing operations | Follow each operation's documented alias contract. Sparse vector and block factors accept an aliased RHS/destination; strided BLAS destinations must not overlap inputs. |

Native factor cleaners are leak guards only; deterministic `close()`/`use` is the lifecycle contract. A strict
allocation policy covers one operation call, not factor construction, native-library initialization, thread
pool startup, or a provider behavior stronger than its declared `AllocationCapability`. OpenBLAS thread
configuration is process-wide even when the selecting `F64Context` is local.
