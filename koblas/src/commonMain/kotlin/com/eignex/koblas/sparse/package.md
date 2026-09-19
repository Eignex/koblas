# Package com.eignex.koblas.sparse

The sparse package contains portable CSC Levels 2 and 3, the sparse Level 1 kernels, and the generic numerical
primitives they share. The public CSC containers live in `com.eignex.koblas`, alongside the dense containers.

[SparseBlas][com.eignex.koblas.sparse.SparseBlas] is the matrix surface: general and symmetric matrix-vector
products, sparse-dense products with the sparse operand on either side, sparse-sparse products into a fresh CSC
result or straight into a dense destination, selected-triangle rank-k products, triangular multiply and solve
for one right-hand side or a block, scaled structural addition, and transpose. All of it is common Kotlin, so
none of it needs an installed library.

[matrixRouteOf][com.eignex.koblas.sparse.SparseBlas.matrixRouteOf] answers what one of those calls executes.
The scheduling is always this library's own portable CSC code, whatever Level 1 kernels the engine selected;
where a column is handed to a Level 1 leaf, the route names the one that width reaches. An engine whose Level 1
kernels are Vector API ones does not thereby execute a vectorised sparse product, and a report that named the
engine would claim exactly that.

Prepared snapshots ([PreparedSparseMatrix][com.eignex.koblas.PreparedSparseMatrix]) own copies of the structure
and the coefficients. A source mutated after preparation does not reach the snapshot, the derived transposed
orientation is published safely for concurrent readers, and no mutable scratch is kept inside: that belongs to
the invocation, which is what lets one snapshot serve several threads with distinct destinations.

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

A stored exact zero is structural and survives, and so does an entry the arithmetic cancels to zero: a product's
pattern is what its operands' patterns meet at, not what its values turn out to be. A position no operand stores
is never visited, so it forms no product and cannot turn an infinity in the other operand into a NaN.
Arithmetic that produces zero does not drop its entry, so a pattern stays stable across updates and compaction
removes only what a caller asks it to remove. A masked
reduction over a selection that is empty returns the identity and reports no position, rather than reporting
position zero, so an empty column is distinguishable from one whose first entry won.
