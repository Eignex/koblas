package com.eignex.koblas.sparse

import com.eignex.koblas.dense.ScalarKernels

/** Exact scalar oracle shared by sparse conformance tests. */
internal object ReferenceSparseBlas :
    SparseBlas by SparseAlgorithms(ScalarKernels, scalarSparseKernelFamilies.indexed, scalarSparseKernelFamilies.panel),
    SparseKernels by ScalarSparseKernels
