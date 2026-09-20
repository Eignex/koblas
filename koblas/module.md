# Module koblas

Portable double-precision dense and sparse BLAS Levels 1–3 for Kotlin Multiplatform.

The immutable [koblas][com.eignex.koblas.koblas] engine uses owned Vector API kernels on a JVM with the
incubator module, and portable Kotlin without it. Kotlin/Native composes optional installed host calls with
portable fallback. Explicit vendor bindings preserve their own arithmetic and overlap contracts;
[openBlas][com.eignex.koblas.vendor.openBlas] returns null on a host with no supported library.

The source runs in one direction. `com.eignex.koblas` holds the containers,
[Workspace][com.eignex.koblas.Workspace] and the convenience extensions, which are written over the BLAS
contracts and reach the default engine to do it. `com.eignex.koblas.dense` and `com.eignex.koblas.sparse`
hold those contracts, the portable scheduling that implements them and the backend kernel seams each
platform fills. `com.eignex.koblas.vendor` holds the explicit host bindings, which a platform default may
compose into dense calls. Nothing below the contracts reads
[koblas][com.eignex.koblas.koblas]: a kernel or a schedule is reached through the engine it was selected
into, never through the default.

[Workspace][com.eignex.koblas.Workspace] reuses temporary storage with bounded retention. It belongs to one
invocation at a time; concurrent calls need separate outputs and scratch. Allocating operations still own
their fresh results. Factorization and solver workflows are outside this artifact.

[BuiltinEngines][com.eignex.koblas.BuiltinEngines], behind the
[KoblasEngineApi][com.eignex.koblas.KoblasEngineApi] opt-in, names exact engines for tests and benchmarks.
[KoblasEngine.routeOf][com.eignex.koblas.KoblasEngine.routeOf] inspects dense and sparse vector and matrix
calls through the same overloaded getter. Routes name the implementation and any fallback; a composed
route identifies multiple components or a choice that depends on values not supplied to the getter.
[Blas.routeOf][com.eignex.koblas.vendor.Blas.routeOf] describes an explicit host call. The host binding
exposes its resolved binary, version and thread evidence.

# Package com.eignex.koblas

Owning dense and sparse containers, borrowed vector views, [Workspace] and high-level arithmetic.
Ordinary use needs only `com.eignex.koblas.*`; [Matrix] and [Vector] are read-only contracts custom types can
implement. Dense matrices are column-major and sparse matrices are validated CSC.

[Matrix.gemm] and [Matrix.gemmInto] dispatch on runtime storage with dense or sparse operands on either side.
Two sparse operands produce CSC storage; other built-in pairings produce dense storage. No sparse operand is
densified to reach a kernel. [PreparedSparseMatrix] owns a snapshot for repeated operations.

# Package com.eignex.koblas.dense

[DenseVectorKernels] supplies Level 1 arithmetic over array windows and strides. [DenseBlas] supplies the
Level 2 and 3 contracts, with shared portable traversal, validation, no-read behavior and alias staging.

The arithmetic inside each window is selected through three backend contracts:

- [DensePanelKernels] executes [PanelWork] and recommends a logical [DensePanelKernels.executionGroup].
- [DenseProductKernels] executes product blocks using a backend-selected register tile and [PackedLayout].
- [DenseTriangularKernels] executes diagonal substitutions and recommends a
  [DenseTriangularKernels.rightHandSideGroup].

Panel groups, SIMD lanes, register tiles, diagonal blocks and cache blocks are independent dimensions.
Structured products reuse the shared schedule over the selected region; their routes report the bodies each
window reaches. Packing and coefficient gathering borrow scoped scratch from [com.eignex.koblas.Workspace].
[PackedMatrix] retains a packed operand for repeated products.

Operands use public containers; transposition and [MatrixStructure] travel as flags. `gemmt` is the
triangle-selected general product Netlib calls `GEMMTR`: full column-major operands and a full-storage
selected output triangle, not compact BLAS packed storage. Explicit host routes report whether the binding
called `cblas_dgemmt` or composed the operation.

# Package com.eignex.koblas.sparse

[SparseBlas] provides portable CSC matrix-vector and matrix-matrix products, symmetric/rank-k operations,
triangular multiply/solve, scaled addition and transpose. Its [SparseBlas.routeOf] reports portable
structural traversal together with the scalar, SIMD or host components actually reached.

Stored zeros participate, cancellation retains an entry, and an absent position is never evaluated. Fresh
results own their arrays with ascending row indices. [com.eignex.koblas.PreparedSparseMatrix] owns copies of
both structure and values; sparse-sparse products may reuse a safely published transpose, while dense products
use the stored snapshot. Mutable scratch belongs to each invocation, so snapshots support concurrent readers.

[SparseKernels] supplies allocation-free indexed arithmetic over validated caller-owned windows. Reductions
admit repeated/unsorted support with contribution semantics; indexed mutations require sorted unique
destinations. [SparseKernels.routeOf] reports each call's actual implementation and fallback.

[SparsePrimitives] provides stateless arithmetic over caller-owned support, marks, accumulators, diagnostics
and output buffers. Pivot selection, permutations, dropping policies, factors and solver state belong to the
consumer. Empty masked reductions return the identity and report no position.
