# Package com.eignex.koblas.sparse

Sparse linear algebra over the CSC [com.eignex.koblas.SparseMatrix], behind three swappable seams that
mirror the dense ones.

- [SparseVectorKernels] — the sparse level-1 tier: a sparse vector against a dense one (`usdot`, `usaxpy`
  in Sparse BLAS terms) or against another sparse one, plus scatter and the reductions. Unlike the dense
  `VectorKernels` there is no length threshold, because the fallback here is an object rather than a
  compiled-in primitive and there is no compile-time kernel to protect.
- [SparseBlas] — the sparse matrix routines. `gemv` in both directions, walking columns, which is what CSC
  stores. Deliberately thin: a sparse `gemm` fills in and is a different algorithm with a different result
  type, so it lands here when something needs it.
- [SparseLapack] — the factorizations, unsymmetric and symmetric. [SparseLapack.factor] returns
  [SparseFactorization], never null: a singular matrix yields a factorization reporting `singular`, matching
  the dense contract. [SparseLapack.cholesky] and [SparseLapack.ldl] are the symmetric pair, and
  [SparseLapack.analyze] is their symbolic half — the phase the unsymmetric factorization cannot have,
  separated because it depends only on the pattern and is therefore reusable across value updates.
- [SparseLinearAlgebra] pairs the two matrix seams. All three are offered through `registerSparseBlas` /
  [com.eignex.koblas.registerBackend] and forced with [com.eignex.koblas.installBackends], resolving as
  [com.eignex.koblas.koblas] and its `sparseVectorKernels`.
- Implementations: [SparseLu], a Markowitz threshold-pivoting `P·B·Q = L·U` that keeps the factors sparse
  instead of filling toward `O(m²)`; and [SparseLdl], an up-looking `A = L·D·Lᵀ` over the elimination tree
  [SparseSymbolic] computes. The two differ in exactly the way their inputs do: an unsymmetric matrix needs
  pivots chosen from the values, a symmetric one is eliminated down its diagonal in the order given.

[SparseFactorization] is an interface rather than a class, which is the one place this deviates from the
dense shape. LAPACK's packed formats are a standard, so a dense [com.eignex.koblas.dense.LuDecomposition]
travels between backends; no sparse solver describes its factors — UMFPACK hands back a `void *`, KLU and
CHOLMOD their own structs — so a seam demanding a concrete type could never admit one.

The containers themselves live in the parent package, alongside the dense ones, because the sealed view
roots require their subtypes in one package.
