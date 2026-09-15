# Package com.eignex.koblas.sparse

The sparse package contains low-level sparse BLAS and kernel building blocks. The public CSC containers and their
high-level operations live in `com.eignex.koblas`, alongside the dense containers. General products support sparse
or direct dense destinations. Symmetric products read exactly one selected CSC triangle, while triangular products
and solves use explicit lower, transpose, unit-diagonal, and side flags. Sparse `syrk` returns either one triangle
of a dense destination or a fresh selected-triangle CSC result; general products do not implicitly mirror it.

`addScaled` and sparse `+`/`-` are root-package algebra extensions. Prepared products retain immutable owned snapshots;
transpose indexing may be cached without introducing descriptors or mutable global selection.

[SparseKernels][com.eignex.koblas.sparse.SparseKernels] also exposes allocation-free raw indexed `dot`, `axpy`,
`scatter`, and stable norm operations. Their independent array windows are validated before arithmetic or
mutation, then dispatched to the engine's scalar, bundled C, or JVM Vector API leaves. Reductions admit repeated
and unsorted support with contribution semantics; indexed mutations require strictly ordered unique destinations.

[SparsePrimitives][com.eignex.koblas.sparse.SparsePrimitives] is a stateless collection of numerical leaves over
caller-owned support, marks, accumulators, diagnostics, and output buffers, including the checked scatter that
reports nonfinite arithmetic and product underflow. It neither owns nor borrows storage;
[Workspace][com.eignex.koblas.Workspace] is reserved for complete higher-level operations that genuinely need
temporary alias staging, packing, accumulation, or multi-result scratch.

The validated solver workflows that were assembled from these leaves are not part of this library. Pivot
selection, permutations, dropping policy, factorization state, and exact arithmetic belong to the consumer that
owns its factors; what remains here is the arithmetic those workflows are built from.

Factorization and basis-solver contracts are not part of this library. A consumer that needs them owns its own
factors on top of these numerical leaves.

A stored exact zero is structural and survives. Arithmetic that produces zero does not drop its entry, so a
pattern stays stable across updates and compaction removes only what a caller asks it to remove. A masked
reduction over a selection that is empty returns the identity and reports no position, rather than reporting
position zero, so an empty column is distinguishable from one whose first entry won.
