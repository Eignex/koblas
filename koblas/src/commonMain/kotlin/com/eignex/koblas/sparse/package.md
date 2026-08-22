# Package com.eignex.koblas.sparse

Sparse linear algebra over the CSC [com.eignex.koblas.core.F64SparseMatrix], behind three swappable seams that
mirror the dense ones.

- [F64SparseVectorKernels] — the sparse level-1 tier: a sparse vector against a dense one (`usdot`, `usaxpy`
  in Sparse BLAS terms) or against another sparse one, plus scatter and the reductions. Unlike the dense
  `F64VectorKernels` there is no length threshold, because the fallback here is an object rather than a
  compiled-in primitive and there is no compile-time kernel to protect.
- [F64SparseBlas] — the sparse matrix routines. `gemv` in both directions, walking columns, which is what CSC
  stores. Deliberately thin: a sparse `gemm` fills in and is a different algorithm with a different result
  type, so it lands here when something needs it.
- [F64SparseLapack] — general sparse LU factorization. [F64SparseLapack.factor] returns
  [F64SparseFactorization], never null: a singular matrix yields a factorization reporting `singular`, matching
  the dense contract. Its factors support both ordinary and transposed solves.
- [F64SparseLinearAlgebra] pairs the two matrix seams. All three are offered through `registerSparseBlas` /
  [com.eignex.koblas.registerBackend] and forced with [com.eignex.koblas.installBackends], resolving as
  [com.eignex.koblas.koblas] and its `sparseVectorKernels`.
- Implementation: [F64SparseLu][com.eignex.koblas.sparse.factorization.lu.F64SparseLu], a Markowitz threshold-pivoting
  `P·B·Q = L·U` that keeps the factors sparse instead of filling toward `O(m²)`.

[F64SparseFactorization] is an interface rather than a class, which is the one place this deviates from the
dense shape. LAPACK's packed formats are a standard, so a dense [com.eignex.koblas.dense.F64LuDecomposition]
travels between backends; no sparse solver describes its factors — UMFPACK hands back a `void *`, KLU and
CHOLMOD their own structs — so a seam demanding a concrete type could never admit one.

The containers themselves live in the parent package, alongside the dense ones, because the sealed view
roots require their subtypes in one package.
