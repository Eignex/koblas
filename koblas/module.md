# Module koblas

Portable double-precision dense and sparse BLAS Levels 1–3 for Kotlin Multiplatform.

The immutable [koblas][com.eignex.koblas.koblas] engine uses owned Vector API kernels on desktop JDK 25 or
later with the incubator module, and portable Kotlin on Android, iOS devices and simulators, and other supported JVMs.
With native access enabled, applicable measured profiles can compose oneMKL for larger JVM matrix calls
or Native vector and matrix calls on Intel Linux x64. Other targets retain portable coverage and can use
explicit vendor bindings or a locally recalculated profile. [BuiltinEngines][com.eignex.koblas.BuiltinEngines], [EngineProfile][com.eignex.koblas.EngineProfile] and
[KoblasEngine.combine][com.eignex.koblas.KoblasEngine.combine] expose the same composition mechanism used by automatic platform selection.

Container conveniences in `com.eignex.koblas` call the BLAS contracts through the default engine.
All engine operations use [KoblasEngine][com.eignex.koblas.KoblasEngine]. Arithmetic implementations share
one [KoblasEngine][com.eignex.koblas.KoblasEngine] contract and declare positive capability sets. Shared
schedules bind panel, product and triangular families before packing or grouping work.
The `vendor` package contains explicit host bindings, which platform engines may compose into eligible dense calls.
Schedules and kernels receive their dependencies from the engine; they do not read the global default.

[Workspace][com.eignex.koblas.Workspace] reuses temporary storage with bounded retention. It belongs to one
invocation at a time; concurrent calls need separate outputs and scratch. Allocating operations still own
their fresh results. Factorization and solver workflows are outside this artifact.

[BuiltinEngines.scalar][com.eignex.koblas.BuiltinEngines.scalar] selects portable Kotlin for every operation
without a host library, JVM flags or changing the default. Every engine uses the same operations and shared
validation, no-read behavior, alias staging and portable schedules. Implementations supply only their declared
arithmetic parts. Norms retain finite-range scaling; rounding, FMA, exceptional results and tied-index choices
follow the selected implementation. Concurrent calls need separate mutable destinations and workspaces.

[BuiltinEngines][com.eignex.koblas.BuiltinEngines] names exact engines for tests and benchmarks, without
annotation opt-ins.
Host backends expose their resolved binary, version, capabilities and single-thread evidence. Missing exports
are absent capabilities and retain portable coverage through the combined engine.

# Package com.eignex.koblas

Owning dense and sparse containers, borrowed vector views, [Workspace] and high-level arithmetic.
Ordinary use needs only `com.eignex.koblas.*`; [Matrix] and [Vector] are read-only contracts custom types can
implement. Dense matrices are column-major and sparse matrices are validated CSC.

[KoblasEngine.gemm] and the `*` operator dispatch on runtime storage with dense or sparse operands on
either side. Two sparse operands produce CSC storage; other built-in pairings produce dense storage. No sparse
operand is densified to reach a kernel. Operators use the default engine.
Dense operations and mixed CSC products address [DenseMatrixRegion] directly, including padded views.

Dense and indexed vector arithmetic, panels, packed products, diagonal substitutions and whole host calls
share [KoblasEngine]. Providers override their supported parts and declare [EngineImplementation] capabilities;
common defaults supply portable arithmetic. [KoblasEngine.combine] binds these parts to an [EngineProfile].
[StandardEngineProfiles] contains the measured configurations used by automatic default selection.

[KoblasEngine.dotColumn] reads a CSC column directly against a dense array without extracting a sparse
vector or rescanning its validated support. Coefficient updates remain visible on each call. Raw indexed
reductions permit repeated/unsorted support; mutations require sorted unique destinations.

# Package com.eignex.koblas.dense

[MatrixStructure] describes the stored triangle and implicit diagonal passed to a whole host call.
Dense matrices and vector windows retain their offsets, column spacing and signed strides. `gemmt` is the
triangle-selected general product Netlib calls `GEMMTR`, over full column-major storage. Missing host exports
retain the shared selected-triangle schedule.

# Package com.eignex.koblas.sparse

[SparsePrimitives] provides stateless arithmetic over caller-owned support, marks, accumulators, diagnostics
and output buffers. Pivot selection, permutations, dropping policies, factors and solver state belong to the
consumer. Empty masked reductions return the identity and report no position.

Sparse Levels 1–3 use [com.eignex.koblas.KoblasEngine]. Traversal stays in validated CSC storage: stored zeros
participate, cancellation retains an entry, and an absent position is never evaluated. Fresh results own their
arrays with ascending row indices; mutable scratch belongs to one invocation.
