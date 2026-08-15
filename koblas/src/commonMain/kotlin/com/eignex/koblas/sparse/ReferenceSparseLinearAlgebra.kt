package com.eignex.koblas.sparse

/** The portable sparse backend, available on every target. */
public object ReferenceSparseLinearAlgebra :
    SparseLinearAlgebra,
    SparseVectorKernels {
    override val name: String get() = "reference"

    override val isPortable: Boolean get() = true
}
