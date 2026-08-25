# Package com.eignex.koblas.sparse.basis

The simplex basis seam: a factorization held across pivots rather than one taken of a matrix.

- [F64BasisSolver] — the seam. A basis is a choice of columns of a matrix fixed for the solver's lifetime,
  named by index, so neither side assembles a square basis per refactorization and a backend that factors
  those columns where they lie never has one to read. It divides the work: the solver owns the factors, the
  pivot order, and the updates, and reports when they are wearing out; a simplex owns pricing, the ratio
  tests, and the refactorization policy that reads those reports.
- [F64IndexedVector] — the carrier the solves read and write, dense values alongside the positions of the
  nonzeros. A simplex iteration on a large model touches a few positions, and a `DoubleArray` would spend
  `O(n)` clearing and rescanning around work that is `O(1)` in the model's size.
- [F64ProductFormBasisSolver] — the portable answer, a sparse LU plus one elementary transform per pivot.
  Its solves run densely between the seam and [com.eignex.koblas.sparse.F64SparseLu]; the seam stays indexed
  so a host binding that solves hypersparsely has nothing to undo.

This is the counterpart of [com.eignex.koblas.sparse.F64BasisFactorization], which factors a basis a caller
hands over whole and answers a column replacement with a superseding factorization. That shape suits a caller
holding one basis; this one suits a caller pivoting through thousands.
