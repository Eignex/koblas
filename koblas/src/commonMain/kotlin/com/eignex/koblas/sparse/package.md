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

  Its transpose, triangle, diagonal, and side choices use the same named Boolean parameters as dense BLAS.
- [F64SparseDecompositions] — the compatibility composition of the selected general LU, Cholesky, LDL, and QR
  roles. [F64SparseDecompositions.factor] is the general LU and
  returns [F64SparseFactorization], never null: a singular matrix yields a factorization reporting
  `singular`, matching the dense contract. [F64SparseDecompositions.cholesky] is `A = L·Lᵀ` for a symmetric
  positive-definite matrix, reading only the lower triangle, and raises where the LU reports, because a
  non-positive pivot says the matrix was not the one the caller described.
  [F64SparseDecompositions.ldl] is `A = L·D·Lᵀ` for a symmetric matrix that need not be definite, and reports
  a zero pivot as singular the way the LU does, a negative one being no failure at all. All three solve in the
  ordinary and the transposed direction, which for the two symmetric ones is the same direction twice.

  [F64SparseDecompositions.qr] is `A = Q·R` of a tall or square matrix, for the least-squares solve
  `min ‖A·x − b‖₂`. It is the one factorization on this seam whose factor is not an [F64SparseFactorization]:
  that type carries a single order and solves between two vectors of it, where an `m×n` QR takes a
  right-hand side of length `m` and answers one of length `n`. [F64SparseQrFactorization] is that type. A
  matrix wider than it is tall is rejected rather than transposed, since what a caller wants from one is the
  minimum-norm solution and that comes from the QR of `Aᵀ`.

  [F64SparseQrFactorization.r], [F64SparseQrFactorization.rank], [F64SparseQrFactorization.columnOrder] and
  [F64SparseQrFactorization.applyQInto] are on it, those being what a QR factorization is. A QR is not
  unique — backends choose different column orderings and different signs — so `columnOrder` names the
  ordering this one chose and the numbers are read against it. Non-uniqueness is a property of the values
  rather than a reason to leave the factors off the type; the conformance tests compare the identities that
  survive it: the least-squares solution, `A·P = Q·R`, `RᵀR = PᵀAᵀAP`, and `Q` returning what `Qᵀ` was
  given.

  `Q` is an operator rather than a matrix. It is `m×m` and dense in general even where `A` and `R` are
  sparse, and libraries hold it as the Householder vectors that build it, so `applyQInto` is the form every
  implementation can answer without materialising something larger than the problem. The SPQR binding asks
  its expert entry point for `R`, the sparse Householder vectors, their coefficients and both permutations in
  one factorization. Those coherent factors outlive the native call as ordinary koblas storage, so applying
  `Q`, reading diagnostics and solving never refactor the input or retain a `cholmod_common`.

  Each kind returns the factor type its own contract names, and every one of them exposes its factors:
  [F64SparseLuFactorization] carries `l`, `u`, the two orderings and the row scaling it was factored under,
  [F64SparseCholeskyFactorization] carries `l` and its ordering, and [F64SparseLdlFactorization] carries `l`,
  `d` and its ordering. The identity each satisfies is written on its interface, in terms of what that
  factorization reports rather than of any particular ordering: a backend that permuted differently satisfies
  the same identity with its own permutation, which is what the conformance helpers check.

  Reading them is materialised on first access and costs a copy out of the library, so a caller who only
  solves never pays. A backend that keeps its factors in a form it cannot hand back raises
  [FactorsNotExposed]; BASICLU and HFactor do, since a basis representation for updating is what they are for.

  Every [F64SparseFactorization] solves either one vector or all columns of a caller-owned dense RHS block.
  The default block path preserves aliasing by staging a column; KLU and CHOLMOD specialize it through one
  native call. An [F64SparseLdlFactorization] additionally exposes its pivot-sign [FactorizationInertia].

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
roles, [F64SparseQr] is least-squares QR, and [F64BasisFactorizations] owns column-replaceable basis factors.
Adding a repeated-pattern or basis provider cannot change the automatic general-LU selection; UMFPACK remains
the accelerated general default when it is available, otherwise the portable implementation does.

[F64RepeatedSparseLu.analyze] returns an explicitly owned [F64SparseLuAnalysis] for one matrix structure. The
analysis privately copies column pointers and row indices, not values, so coefficient arrays can change between
numeric factorizations. A different structure raises [IllegalArgumentException] before refactoring; this is
distinct from numerical singularity. Numeric factors stay caller-owned and must be closed before the analysis.
KLU reuses the symbolic ordering through this typed capability, so a same-pattern loop needs no concrete backend
cast.

An explicit [com.eignex.koblas.F64ContextBuilder] can select any provider for any role it implements. Use
[com.eignex.koblas.F64Capabilities] with [com.eignex.koblas.backendNamed] to retrieve an exact discovered
provider without a concrete implementation cast, and [com.eignex.koblas.capability] to retrieve
the provider held by a context. A backend is reached by name through the capability it fills; there is no
untyped lookup, because a backend now offers the roles it implements rather than the seam it happens to
satisfy.

Row equilibration and a drop tolerance are policy a backend's constructor settles, beside the settings that
already say how to scale; the portable implementation settles them in [F64ReferenceSparseDecompositions].

A library filling one kind and not another is the ordinary case rather than the exception, since most of
these are unsymmetric LU and nothing else. Such a binding answers the rest portably through
[F64SparseDecompositionsAdapter][com.eignex.koblas.sparse.host.F64SparseDecompositionsAdapter], which is the
same portable implementation it uses for unsupported operations.

CHOLMOD supplies the symmetric routines used by the SuiteSparse providers, and SPQR the QR. Those
capabilities compete only with other providers of the same role, independently of which provider fills
general or repeated-pattern LU.

- LU: [F64SparseMarkowitzLu][com.eignex.koblas.sparse.factorization.lu.F64SparseMarkowitzLu], a
  Markowitz threshold-pivoting `P·B·Q = L·U` that keeps the factors sparse instead of filling toward `O(m²)`.
- Cholesky:
  [F64SparseUpLookingCholesky][com.eignex.koblas.sparse.factorization.cholesky.F64SparseUpLookingCholesky],
  an up-looking `A = L·Lᵀ` over the elimination tree, in the ordering the matrix arrives in.
- QR: [F64SparseHouseholderQr][com.eignex.koblas.sparse.factorization.qr.F64SparseHouseholderQr], Householder
  reflections over the elimination tree of `AᵀA`, in the ordering the matrix arrives in. `Q` is held as the
  reflections rather than formed, since it is `m×m` and dense in general where `A` and `R` are sparse.

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
