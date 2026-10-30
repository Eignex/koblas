# Module koblas

Portable double-precision dense and sparse BLAS Levels 1–3 for Kotlin Multiplatform.

The immutable [koblas][com.eignex.koblas.koblas] engine uses owned Vector API kernels on desktop JDK 25 or
later with the incubator module, and portable Kotlin on Android, iOS devices and simulators, and other supported JVMs.
With native access enabled, the JVM can also compose oneMKL for eligible Level 3 calls on Intel x64 Linux and Windows.
Linux, Windows and macOS Kotlin/Native compose optional installed host calls with portable fallback. Windows
reuses the Linux routing thresholds and vendor preference order, with separately installed oneMKL, AOCL,
ArmPL or OpenBLAS DLLs. Windows Arm64 is available through the JVM; Kotlin/Native supports Windows x64.
Explicit vendor bindings preserve their own arithmetic and overlap contracts;
[openBlas][com.eignex.koblas.vendor.openBlas] returns null on Android, JDKs below 25, or a host with no
supported library. The JVM publication is one multi-release JAR with no bundled native code.

Container conveniences in `com.eignex.koblas` call the BLAS contracts through the default engine.
The `dense` and `sparse` packages expose operations and useful raw array arithmetic; scheduling and
panel/product/triangular implementation interfaces are internal.
The `vendor` package contains explicit host bindings, which platform engines may compose into eligible dense calls.
Schedules and kernels receive their dependencies from the engine; they do not read the global default.

[Workspace][com.eignex.koblas.Workspace] reuses temporary storage with bounded retention. It belongs to one
invocation at a time; concurrent calls need separate outputs and scratch. Allocating operations still own
their fresh results. Factorization and solver workflows are outside this artifact.

[KoblasEngine.portable][com.eignex.koblas.KoblasEngine.portable] creates an immutable application engine
using portable Kotlin for every operation, without an opt-in, host library or JVM flags. Call that instance
directly; container conveniences continue to use the platform default. Each routine permits normal BLAS
rounding, FMA and reduction-order latitude. Norms retain finite-range scaling; unspecified exceptional
results and tied-index choices belong to the selected implementation. Concurrent calls need separate
mutable destinations and workspaces.

[BuiltinEngines][com.eignex.koblas.BuiltinEngines] names exact engines for tests and benchmarks, without
annotation opt-ins.
[Blas.routeOf][com.eignex.koblas.vendor.Blas.routeOf] describes an explicit host call. The host binding
exposes its resolved binary, version and thread evidence.

# Package com.eignex.koblas

Owning dense and sparse containers, borrowed vector views, [Workspace] and high-level arithmetic.
Ordinary use needs only `com.eignex.koblas.*`; [Matrix] and [Vector] are read-only contracts custom types can
implement. Dense matrices are column-major and sparse matrices are validated CSC.

[KoblasEngine.gemm] and the `*` operator dispatch on runtime storage with dense or sparse operands on
either side. Two sparse operands produce CSC storage; other built-in pairings produce dense storage. No sparse
operand is densified to reach a kernel. Operators use the default engine.
Dense operations and mixed CSC products address [DenseMatrixRegion] directly, including padded views.

# Package com.eignex.koblas.dense

[DenseVectorKernels] supplies Level 1 arithmetic over array windows and strides. [DenseBlas] supplies the
Level 2 and 3 contracts, with shared portable traversal, validation, no-read behavior and alias staging.

Shared internal kernels supply panel arithmetic, packed product tiles and diagonal substitutions.
Transient packing and coefficient gathering borrow scoped scratch from [com.eignex.koblas.Workspace].

Dense matrix operands use [com.eignex.koblas.DenseMatrixRegion] and vectors use
[com.eignex.koblas.DenseVector], preserving offsets, leading dimensions and signed strides; transposition and [MatrixStructure] travel as flags. `gemmt` is the
triangle-selected general product Netlib calls `GEMMTR`: full column-major operands and a full-storage
selected output triangle, not compact BLAS packed storage. A host binding without `cblas_dgemmt` composes
the operation from `gemm`.

# Package com.eignex.koblas.sparse

[SparseBlas] provides portable CSC matrix-vector and matrix-matrix products, symmetric/rank-k operations,
triangular multiply/solve, scaled addition and transpose, over the engine's Level 1 kernels.

Stored zeros participate, cancellation retains an entry, and an absent position is never evaluated. Fresh
results own their arrays with ascending row indices. Mutable scratch belongs to each invocation.

[SparseKernels] supplies allocation-free indexed arithmetic over validated caller-owned windows. Reductions
admit repeated/unsorted support with contribution semantics; indexed mutations require sorted unique
destinations.
[SparseKernels.dotColumn] reads a selected CSC column against a dense array, reusing the matrix's validated
support without extracting a vector or rescanning indices. Coefficient updates remain visible on every call.

[SparsePrimitives] provides stateless arithmetic over caller-owned support, marks, accumulators, diagnostics
and output buffers. Pivot selection, permutations, dropping policies, factors and solver state belong to the
consumer. Empty masked reductions return the identity and report no position.
