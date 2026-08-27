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

  [F64SparseBlas.prepare] takes an immutable CSC snapshot for repeated products. A CHOLMOD implementation
  retains its native descriptor until the returned [F64PreparedSparseMatrix] is closed, so iterative methods
  do not recopy column pointers, row indices, and values on every multiply. The prepared-operation gates are
  separate from the setup-inclusive one-shot gate. Handles reject calls after close and require external
  serialization when shared between threads.

  Its transpose, triangle, diagonal, and side choices have the same typed
  [com.eignex.koblas.dense.Transpose], [com.eignex.koblas.dense.Uplo], [com.eignex.koblas.dense.Diag], and
  [com.eignex.koblas.dense.Side] overloads as dense BLAS. Boolean forms remain source compatible.
- [F64SparseDecompositions] — the compatibility composition of the selected general LU, Cholesky, and LDL
  roles. [F64SparseDecompositions.factor] is the general LU and
  returns [F64SparseFactorization], never null: a singular matrix yields a factorization reporting
  `singular`, matching the dense contract. [F64SparseDecompositions.cholesky] is `A = L·Lᵀ` for a symmetric
  positive-definite matrix, reading only the lower triangle, and raises where the LU reports, because a
  non-positive pivot says the matrix was not the one the caller described.
  [F64SparseDecompositions.ldl] is `A = L·D·Lᵀ` for a symmetric matrix that need not be definite, and reports
  a zero pivot as singular the way the LU does, a negative one being no failure at all. All three solve in the
  ordinary and the transposed direction, which for the two symmetric ones is the same direction twice.

  Every [F64SparseFactorization] solves either one vector or all columns of a caller-owned dense RHS block.
  The default block path preserves aliasing by staging a column; KLU and CHOLMOD specialize it through one
  native call. [F64SparseFactorization.report] samples an extensible [F64SparseFactorizationReport]: null means
  unavailable for optional fields, while zero remains a valid reported value. Portable LU exposes its row and
  column permutations and ordering; portable LDL additionally reports inertia.

  The sparse `ldl` is not the sparse counterpart of the dense one, whatever the shared name suggests. The
  dense one pivots for stability; neither the portable sparse one nor any library behind this seam does, the
  permutation being chosen to limit fill with nothing reordering on the numbers. It is the factorization for a
  quasi-definite matrix, which is what an interior point method's KKT system is, and a caller who cannot
  promise that wants [F64SparseDecompositions.factor], whose pivoting is numerical.
- [F64BasisFactorization] — a sparse LU factorization of a simplex basis. It retains the basis matrix and
  can produce the factorization after one column replacement, which may be any column at all. BASICLU
  answers it; [F64RefactoringBasisFactorization] wraps any other backend at the cost of a factorization per
  replacement. A caller pivoting a basis named by index into a fixed matrix wants
  [F64BasisSolver][com.eignex.koblas.sparse.basis.F64BasisSolver] instead, on its own backend half.
- [F64SparseLinearAlgebra] pairs the matrix seams and exposes the sparse-vector kernels alongside them.
  Backends may implement either matrix half; [com.eignex.koblas.registerBackend] ranks each independently,
  while [com.eignex.koblas.installBackends] supplies all three through [com.eignex.koblas.koblas].

Sparse libraries are specialized: KLU wants a circuit pattern it can factor repeatedly, UMFPACK an
unstructured system, BASICLU a basis whose columns are replaced one at a time, and
[com.eignex.koblas.sparse.basis.F64BasisSolver] a basis pivoted thousands of times. The registry therefore
ranks providers only within semantic roles. [F64GeneralSparseLu] is ordinary pivoting LU,
[F64RepeatedSparseLu] adds same-pattern refactorization, [F64SparseCholesky] and [F64SparseLdl] are symmetric
roles, and [F64BasisFactorizations] owns column-replaceable basis factors. Adding a repeated-pattern or basis
provider cannot change the automatic general-LU selection; UMFPACK remains the accelerated general default
when it is available, otherwise the portable implementation does.

[F64RepeatedSparseLu.analyze] returns an explicitly owned [F64SparseLuAnalysis] for one [F64SparsePattern].
The pattern copies column pointers and row indices, not values, so coefficient arrays can change between
numeric factorizations. A different structure raises [IncompatibleSparsePatternException] before refactoring;
this is distinct from numerical singularity. Numeric factors stay caller-owned and must be closed before the
analysis. KLU reuses the symbolic ordering through this typed capability, so a same-pattern loop needs no
concrete backend cast.

An explicit [com.eignex.koblas.F64ContextBuilder] can select any provider for any role it implements. Use
[com.eignex.koblas.F64Capabilities] with [com.eignex.koblas.backendNamed] to retrieve an exact discovered
provider without a concrete implementation cast, and [com.eignex.koblas.capability] to retrieve
the provider held by a context. [com.eignex.koblas.sparseDecompositionsNamed] remains as a compatibility API
for backend-specific surface.

Row equilibration and a drop tolerance are policy a backend's constructor settles, beside the settings that
already say how to scale; the portable implementation settles them in [F64ReferenceSparseDecompositions].

A library filling one kind and not another is the ordinary case rather than the exception, since most of
these are unsymmetric LU and nothing else. Such a binding answers the rest portably through
[F64SparseDecompositionsAdapter][com.eignex.koblas.sparse.host.F64SparseDecompositionsAdapter], which is the
same fallback it already uses below its own size gate.

CHOLMOD supplies the symmetric routines used by the SuiteSparse providers. Those capabilities compete only
with other Cholesky or LDL providers, independently of which provider fills general or repeated-pattern LU.

- LU: [F64SparseLuFactorization][com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization], a
  Markowitz threshold-pivoting `P·B·Q = L·U` that keeps the factors sparse instead of filling toward `O(m²)`.
- Cholesky:
  [F64SparseCholeskyFactorization][com.eignex.koblas.sparse.factorization.cholesky.F64SparseCholeskyFactorization],
  an up-looking `A = L·Lᵀ` over the elimination tree, in the ordering the matrix arrives in.

[F64SparseFactorization] is an interface rather than a class, which is the one place this deviates from the
dense shape. LAPACK's packed formats are a standard, so a dense [com.eignex.koblas.dense.F64LuDecomposition]
travels between backends; no sparse solver describes its factors — UMFPACK hands back a `void *`, KLU and
CHOLMOD their own structs — so a seam demanding a concrete type could never admit one.

It is also [AutoCloseable][kotlin.AutoCloseable]. A native factor owns the opaque objects the library handed
back and should be held in `use` or closed explicitly after its final solve. Close is idempotent and waits for
an in-flight native call before releasing; a cleaner follows the same one-shot path only if the caller loses
the factor first. Portable factors have no external resource and close as a no-op. A closed native factor
keeps its dimension and singular status but rejects solves and factor-statistic reads.

The containers themselves live in the parent package, alongside the dense ones, because the sealed view
roots require their subtypes in one package.
