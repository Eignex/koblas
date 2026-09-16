# Package com.eignex.koblas.sparse

The sparse package contains sparse Level 1 kernels and the generic numerical primitives they share. The public
CSC containers live in `com.eignex.koblas`, alongside the dense containers. Sparse matrix products are not part
of this library: a vector against a matrix column, or against a dense vector, is Level 1 work on a stored
support, and anything above that belongs to the consumer that owns the algorithm.

[SparseKernels][com.eignex.koblas.sparse.SparseKernels] exposes allocation-free raw indexed `dot`, `axpy`,
`scatter`, and stable norm operations. Their independent array windows are validated before arithmetic or
mutation, then dispatched to the engine's scalar or JVM Vector API leaves. Reductions admit repeated
and unsorted support with contribution semantics; indexed mutations require strictly ordered unique destinations.

[routeOf][com.eignex.koblas.sparse.SparseKernels.routeOf] answers where one such call executes, from the
decision the call itself makes. A selection is not evidence of what ran: a Vector API selection answers a call
below its crossover, an operation it never vectorised, and a host whose indexed loads or stores it cannot use by
handing the whole call to the scalar kernels, and a measurement that reads only the selection name publishes
those as vector timings.

[SparsePrimitives][com.eignex.koblas.sparse.SparsePrimitives] is a stateless collection of numerical leaves over
caller-owned support, marks, accumulators, diagnostics, and output buffers, including the checked scatter that
reports nonfinite arithmetic and product underflow. It neither owns nor borrows storage: every buffer a
routine writes through is one the caller passed in, which is what lets these run inside a hot loop without
the collector noticing.

The validated solver workflows assembled from these leaves are not part of this library, and neither are
factorization or basis-solver contracts. Pivot selection, permutations, dropping policy, factorization state,
and exact arithmetic belong to the consumer that owns its factors; what remains here is the arithmetic those
workflows are built from.

A stored exact zero is structural and survives. Arithmetic that produces zero does not drop its entry, so a
pattern stays stable across updates and compaction removes only what a caller asks it to remove. A masked
reduction over a selection that is empty returns the identity and reports no position, rather than reporting
position zero, so an empty column is distinguishable from one whose first entry won.
