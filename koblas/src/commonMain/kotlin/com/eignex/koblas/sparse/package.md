# Package com.eignex.koblas.sparse

Sparse linear algebra over the CSC [com.eignex.koblas.core.F64SparseMatrix], behind three swappable seams that
mirror the dense ones.

- [F64SparseKernels] — the sparse level-1 tier: a sparse vector against a dense one (`usdot`, `usaxpy`
  in Sparse BLAS terms) or against another sparse one, plus scatter and the reductions. Unlike the dense
  `F64Kernels` there is no length threshold, because the fallback here is an object rather than a
  compiled-in primitive and there is no compile-time kernel to protect.
- [F64SparseBlas] — the sparse matrix routines. `gemv` in both directions, walking columns, which is what CSC
  stores. Deliberately thin: a sparse `gemm` fills in and is a different algorithm with a different result
  type, so it lands here when something needs it.
- [F64SparseLu] — general sparse LU factorization. [F64SparseLu.factor] returns
  [F64SparseFactorization], never null: a singular matrix yields a factorization reporting `singular`, matching
  the dense contract. Its factors support both ordinary and transposed solves.
- [F64BasisFactorization] — a sparse LU factorization of a simplex basis. It retains the basis matrix and
  can produce the factorization after one column replacement, which may be any column at all. A caller
  pivoting a basis named by index into a fixed matrix wants
  [F64BasisSolver][com.eignex.koblas.sparse.basis.F64BasisSolver] instead, on its own backend half.
- [F64SparseLinearAlgebra] pairs the matrix seams and exposes the sparse-vector kernels alongside them.
  Backends may implement either matrix half; [com.eignex.koblas.registerBackend] ranks each independently,
  while [com.eignex.koblas.installBackends] supplies all three through [com.eignex.koblas.koblas].
- Implementation: [F64SparseLuFactorization][com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization], a Markowitz threshold-pivoting
  `P·B·Q = L·U` that keeps the factors sparse instead of filling toward `O(m²)`.

[F64SparseFactorization] is an interface rather than a class, which is the one place this deviates from the
dense shape. LAPACK's packed formats are a standard, so a dense [com.eignex.koblas.dense.F64LuDecomposition]
travels between backends; no sparse solver describes its factors — UMFPACK hands back a `void *`, KLU and
CHOLMOD their own structs — so a seam demanding a concrete type could never admit one.

The containers themselves live in the parent package, alongside the dense ones, because the sealed view
roots require their subtypes in one package.
