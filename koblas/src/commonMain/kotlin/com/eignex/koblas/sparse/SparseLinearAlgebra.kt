package com.eignex.koblas.sparse

import com.eignex.koblas.koblas

/** Sparse BLAS with the active sparse-vector kernels used by its surrounding operations. */
public interface SparseLinearAlgebra : SparseBlas {
    /** The sparse vector kernels used by operations around this matrix half. */
    public val sparseKernels: SparseKernels get() = koblas.sparseKernels
}
