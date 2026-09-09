package com.eignex.koblas.sparse

import com.eignex.koblas.dense.ScalarKernels
import com.eignex.koblas.dense.ScalarPanelKernels

/** Exact scalar oracle shared by sparse conformance tests. */
internal object ReferenceSparseLinearAlgebra :
    SparseBlas by SparseAlgorithms(ScalarKernels, ScalarPanelKernels),
    SparseKernels by ScalarSparseKernels {
    override val name: String = "reference"
}
