# Package com.eignex.koblas.sparse

Sparse linear algebra over the CSC [com.eignex.koblas.SparseMatrix].

- [SparseLu] — a Markowitz threshold-pivoting `P·B·Q = L·U` factorization that keeps the factors sparse
  instead of filling toward `O(m²)`, with `O(nnz)` triangular solves in both directions and a
  determinant.
- [EtaBasis] — the product-form-of-the-inverse: a base factorization plus a growing list of rank-one eta
  updates, which is how a revised simplex avoids refactorizing every iteration.

The containers themselves live in the parent package, alongside the dense ones, because the sealed view
roots require their subtypes in one package.
