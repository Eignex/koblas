package com.eignex.koblas.sparse

public object ReferenceSparseLinearAlgebra :
    SparseLinearAlgebra,
    SparseVectorKernels {
    override val name: String get() = "reference"
}
