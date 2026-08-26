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

- [F64BasisSolvers] — the backend half. It is separate from [com.eignex.koblas.sparse.F64SparseLu] because
  the two basis contracts want different libraries, and held in one seam the strongest at either would take
  the other from whoever was strongest there.

This is the counterpart of [com.eignex.koblas.sparse.F64BasisFactorization], which factors a basis a caller
hands over whole and answers a column replacement with a superseding factorization. The difference is not
that one supersedes the other. That one takes *any* entering column, so it wants a backend carrying the
basis independently of a matrix, which is BASICLU and is what an interior point method's basis
preconditioner asks for. This one names the entering column by index into a matrix fixed for the solver's
lifetime, which is what lets a backend keep the factors where the columns lie and solve hypersparsely, and
is what a simplex pivoting through thousands of bases asks for. A backend reading its columns out of a
matrix by index cannot answer the first: the update would go through, but the refactorization behind it
would have no column to rebuild from.
