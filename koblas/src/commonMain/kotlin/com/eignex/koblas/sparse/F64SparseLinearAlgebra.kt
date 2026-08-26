package com.eignex.koblas.sparse

import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.basis.F64BasisSolvers

/** The sparse matrix halves, with the active sparse-vector kernels used by their surrounding operations. */
public interface F64SparseLinearAlgebra :
    F64SparseBlas,
    F64SparseDecompositions,
    F64BasisSolvers {
    /** The sparse vector kernels used by operations around these matrix halves. */
    public val sparseKernels: F64SparseKernels get() = koblas.sparseKernels
}
