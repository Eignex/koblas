package com.eignex.koblas.dense

import com.eignex.koblas.F64Context
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.koblas

/**
 * Portable pure-Kotlin backend, correct on every target with no native dependency, and the semantic
 * reference a native backend is validated against. The routines themselves live in [F64ReferenceBlas] and
 * [F64ReferenceDecompositions]; this composes them so one object satisfies the whole dense seam.
 *
 * @param configured the kernels the inner loops use, or null to follow the [F64Context] default.
 */
public class F64ReferenceBackend(private val configured: F64Kernels? = null) :
    F64LinearAlgebra,
    F64Blas by F64ReferenceBlas(configured),
    F64Decompositions by F64ReferenceDecompositions(configured) {
    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    /** koblas's own implementation, so it runs anywhere koblas does. */
    override val isAvailable: Boolean get() = true

    override val priority: Int get() = 0

    /** This backend's kernels, or the process default when it was given none. */
    override val kernels: F64Kernels get() = configured ?: koblas.kernels
}

/** The shared portable backend, the fallback every seam resolves to when nothing else is registered. */
public val F64ReferenceLinearAlgebra: F64ReferenceBackend = F64ReferenceBackend()
