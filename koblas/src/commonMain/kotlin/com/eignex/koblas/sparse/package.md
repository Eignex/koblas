# Package com.eignex.koblas.sparse

The sparse package contains validated CSC containers and sparse BLAS levels 1–3. General products support sparse
or direct dense destinations. Symmetric products read exactly one selected CSC triangle, while triangular products
and solves use explicit lower, transpose, unit-diagonal, and side flags. Sparse `syrk` returns either one triangle
of a dense destination or a fresh selected-triangle CSC result; general products do not implicitly mirror it.

`addScaled` and sparse `+`/`-` are sparse algebra extensions. Prepared products retain immutable owned snapshots;
transpose indexing may be cached without introducing descriptors or mutable global selection.

[SparseKernels][com.eignex.koblas.sparse.SparseKernels] also exposes allocation-free raw indexed `dot`, `axpy`,
`scatter`, and stable norm operations. Their independent array windows are validated before arithmetic or
mutation, then dispatched to the context's scalar, bundled C, or JVM Vector API leaves. Reductions admit repeated
and unsorted support with contribution semantics; indexed mutations require strictly ordered unique destinations.

[SparseSlices][com.eignex.koblas.sparse.SparseSlices] is a stateless collection of validated operations over
caller-owned support, marks, accumulators, diagnostics, and output buffers. It includes ordered checked arithmetic
and direct touched-support clearing. It neither owns nor borrows storage; [Workspace][com.eignex.koblas.Workspace]
is reserved for complete higher-level operations that genuinely need temporary alias staging, packing,
transposition, accumulation, or multi-result scratch. Pivot selection, permutations, dropping policy,
factorization state, and exact arithmetic remain outside these numerical leaves.

Factorization and basis-solver contracts live in the optional JVM `koblas-hfactor` artifact. The multiplatform
BLAS module neither loads nor detects HFactor.
