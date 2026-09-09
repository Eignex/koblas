package com.eignex.koblas.sparse

import com.eignex.koblas.dense.ScalarKernels

/** Exact scalar oracle shared by sparse conformance tests. */
internal object ReferenceSparseLinearAlgebra :
    SparseBlas by SparseAlgorithms(ScalarKernels),
    SparseKernels by ScalarSparseKernels {
    override val name: String = "reference"
}
