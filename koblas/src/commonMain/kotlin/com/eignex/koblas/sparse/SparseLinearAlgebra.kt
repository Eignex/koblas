package com.eignex.koblas.sparse

import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.basis.F64BasisSolvers

/** The sparse matrix halves, with the active sparse-vector kernels used by their surrounding operations. */
public interface SparseLinearAlgebra :
    SparseBlas,
    SparseLapack,
    F64BasisSolvers {
    /** The sparse vector kernels used by operations around these matrix halves. */
    public val sparseKernels: SparseKernels get() = koblas.sparseKernels
}
