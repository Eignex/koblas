package com.eignex.koblas.internal.sparse

/** Scalar indexed arithmetic is the interface default and the finite-data conformance oracle. */
internal object ScalarIndexedSparseKernels : IndexedSparseKernels {
    override val name: String = "scalar"
}
