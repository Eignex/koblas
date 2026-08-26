# Package com.eignex.koblas.sparse

Sparse linear algebra over the CSC [com.eignex.koblas.core.F64SparseMatrix], behind three swappable seams that
mirror the dense ones.

- [F64SparseKernels] — the sparse level-1 tier: a sparse vector against a dense one (`usdot`, `usaxpy`
  in Sparse BLAS terms) or against another sparse one, the scatter and gather pair (`ussc`, `usga`, `usgz`)
  that moves entries between the two representations, and the reductions. Unlike the dense
  `F64Kernels` there is no length threshold, because the fallback here is an object rather than a
  compiled-in primitive and there is no compile-time kernel to protect.
- [F64SparseBlas] — the sparse matrix routines. `gemv` and `trsv` in both directions, walking columns, which
  is what CSC stores; `gemm` and `trsm` over several right-hand sides at once, from either side; the
  sparse-times-sparse product; and `transpose`, which is also the CSC-to-CSR conversion.

  The product of two sparse matrices and the transpose are the two routines here that return their result
  instead of filling a destination. Not for want of a type, since [com.eignex.koblas.core.F64SparseMatrix] is
  the type either would fill: what a sparse product discovers is its own pattern, so there is no destination
  to hand in before the multiplication has run and no `beta · C` to accumulate into. Everything else on this
  seam keeps the BLAS shape, where the caller owns the memory.
- [F64SparseDecompositions] — the sparse factorizations, one seam over all of them as the dense
  [com.eignex.koblas.dense.F64Decompositions] is. [F64SparseDecompositions.factor] is the general LU and
  returns [F64SparseFactorization], never null: a singular matrix yields a factorization reporting
  `singular`, matching the dense contract. [F64SparseDecompositions.cholesky] is `A = L·Lᵀ` for a symmetric
  positive-definite matrix, reading only the lower triangle, and raises where the LU reports, because a
  non-positive pivot says the matrix was not the one the caller described. Both factorizations solve in the
  ordinary and the transposed direction, which for the Cholesky is the same direction twice.
- [F64BasisFactorization] — a sparse LU factorization of a simplex basis. It retains the basis matrix and
  can produce the factorization after one column replacement, which may be any column at all. BASICLU
  answers it; [F64RefactoringBasisFactorization] wraps any other backend at the cost of a factorization per
  replacement. A caller pivoting a basis named by index into a fixed matrix wants
  [F64BasisSolver][com.eignex.koblas.sparse.basis.F64BasisSolver] instead, on its own backend half.
- [F64SparseLinearAlgebra] pairs the matrix seams and exposes the sparse-vector kernels alongside them.
  Backends may implement either matrix half; [com.eignex.koblas.registerBackend] ranks each independently,
  while [com.eignex.koblas.installBackends] supplies all three through [com.eignex.koblas.koblas].

The sparse backends differ from the dense ones in a way the ranking cannot express. A dense binding is the
same routine computed faster, so ordering them by [com.eignex.koblas.Backend.priority] and taking the
strongest is the whole story. The sparse libraries are specialised instead: KLU wants a circuit pattern it
can factor repeatedly, UMFPACK an unstructured system, BASICLU a basis whose columns are replaced one at a
time, [com.eignex.koblas.sparse.basis.F64BasisSolver] a basis pivoted thousands of times. Which is best is
a property of the matrix, not of a ranking, and one process can want two at once.

So [com.eignex.koblas.koblas] hands out the strongest offer per half, which is the right default, and
[com.eignex.koblas.sparseDecompositionsNamed] hands out a particular one for a caller that knows which it wants.

For the same reason [F64SparseDecompositions] carries only what a factorization of each kind universally
does. What one library has and the others do not is a routine on that library: KLU's `refactor`, which
reuses a symbolic analysis across matrices of one pattern; BASICLU's `factorBasis`; HFactor's `basisSolver`,
which has a half of its own because the portable backend answers it too. Row equilibration and a drop
tolerance are policy a backend's constructor settles, beside the settings that already say how to scale; the
portable half settles them in [F64ReferenceSparseDecompositions], whose `Default` is what the seam falls
back to.

A library filling one kind and not another is the ordinary case rather than the exception, since most of
these are unsymmetric LU and nothing else. Such a binding answers the rest portably through
[F64SparseDecompositionsAdapter][com.eignex.koblas.sparse.host.F64SparseDecompositionsAdapter], which is the
same fallback it already uses below its own size gate.

- LU: [F64SparseLuFactorization][com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization], a
  Markowitz threshold-pivoting `P·B·Q = L·U` that keeps the factors sparse instead of filling toward `O(m²)`.
- Cholesky:
  [F64SparseCholeskyFactorization][com.eignex.koblas.sparse.factorization.cholesky.F64SparseCholeskyFactorization],
  an up-looking `A = L·Lᵀ` over the elimination tree, in the ordering the matrix arrives in.

[F64SparseFactorization] is an interface rather than a class, which is the one place this deviates from the
dense shape. LAPACK's packed formats are a standard, so a dense [com.eignex.koblas.dense.F64LuDecomposition]
travels between backends; no sparse solver describes its factors — UMFPACK hands back a `void *`, KLU and
CHOLMOD their own structs — so a seam demanding a concrete type could never admit one.

The containers themselves live in the parent package, alongside the dense ones, because the sealed view
roots require their subtypes in one package.
