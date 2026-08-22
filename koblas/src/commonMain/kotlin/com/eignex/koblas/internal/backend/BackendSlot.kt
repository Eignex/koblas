package com.eignex.koblas.internal.backend

/** The halves of the seam a backend can implement. */
public enum class BackendSlot {
    /** Dense vector-vector routines. */
    F64VectorKernels,

    /** Dense matrix routines. */
    F64Blas,

    /** Dense factorizations. */
    F64Lapack,

    /** Sparse vector-vector routines. */
    F64SparseVectorKernels,

    /** Sparse matrix routines. */
    F64SparseBlas,

    /** Sparse factorizations. */
    F64SparseLapack,
}
