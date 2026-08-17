package com.eignex.koblas.dense

import com.eignex.koblas.BackendNames
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.koblas

/**
 * Portable pure-Kotlin backend, correct on every target with no native dependency, and the semantic
 * reference a native backend is validated against. The routines themselves live in [ReferenceBlas] and
 * [ReferenceLapack]; this composes them so one object satisfies the whole dense seam.
 *
 * @param kernels the vector kernels the inner loops use, or null to follow the [KoblasContext] default.
 */
public class ReferenceBackend(private val kernels: VectorKernels? = null) :
    LinearAlgebra,
    Blas by ReferenceBlas(kernels),
    Lapack by ReferenceLapack(kernels) {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    override val priority: Int get() = 0

    /** This backend's kernels, or the process default when it was given none. */
    override val vectorKernels: VectorKernels get() = kernels ?: koblas.vectorKernels
}

/** The shared portable backend, the fallback every seam resolves to when nothing else is registered. */
public val ReferenceLinearAlgebra: ReferenceBackend = ReferenceBackend()
