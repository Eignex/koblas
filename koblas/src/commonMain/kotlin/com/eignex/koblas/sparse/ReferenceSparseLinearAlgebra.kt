package com.eignex.koblas.sparse

import com.eignex.koblas.BackendNames

/** The portable sparse backend, available on every target. */
public object ReferenceSparseLinearAlgebra :
    SparseLinearAlgebra,
    SparseVectorKernels {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true
}
