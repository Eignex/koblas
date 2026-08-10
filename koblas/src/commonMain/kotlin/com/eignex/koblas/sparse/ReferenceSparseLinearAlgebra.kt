package com.eignex.koblas.sparse

/**
 * The portable sparse backend: every routine in Kotlin, no host dependency, and the semantic reference a
 * host backend is validated against.
 *
 * Every operation is an interface default, so this object overrides nothing — the algorithms live on the
 * seams where a backend can replace them one at a time.
 */
public object ReferenceSparseLinearAlgebra :
    SparseLinearAlgebra,
    SparseVectorKernels {
    override val name: String get() = "reference"
}
