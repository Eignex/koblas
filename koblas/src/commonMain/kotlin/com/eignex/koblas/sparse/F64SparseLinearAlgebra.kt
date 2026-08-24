package com.eignex.koblas.sparse

import com.eignex.koblas.koblas

/** The sparse matrix halves, with the active sparse-vector kernels used by their surrounding operations. */
public interface F64SparseLinearAlgebra :
    F64SparseBlas,
    F64SparseLu {
    /** The sparse vector kernels used by operations around these matrix halves. */
    public val sparseVectorKernels: F64SparseVectorKernels get() = koblas.sparseVectorKernels
}
