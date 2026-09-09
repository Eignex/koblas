# Package com.eignex.koblas.sparse

The sparse package contains CSC containers, sparse BLAS levels 1–3, sparse-dense and sparse-sparse
products, transpose, preparation, triangular multiplication and triangular solves.

Portable sparse Cholesky, LDL, LU, QR, symbolic analysis and refactorization have been removed. The
remaining [GeneralSparseLu], [SparseFactorization], and [SparseLuFactorization] contracts serve the separately
packaged HFactor backend. They have no portable implementation
and will disappear with HFactor in a later change.
