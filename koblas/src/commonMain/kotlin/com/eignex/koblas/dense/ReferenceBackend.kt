package com.eignex.koblas.dense

import com.eignex.koblas.F64RebindableBackend
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.koblas

/**
 * Portable pure-Kotlin backend, correct on every target with no native dependency, and the semantic
 * reference a native backend is validated against. The routines themselves live in [F64PortableBlas].
 *
 * @param configured the kernels the inner loops use, or null to follow the [KoblasContext] default.
 */
public class ReferenceBackend(private val configured: Kernels? = null) :
    F64RebindableBackend,
    Blas by F64PortableBlas(configured) {
    override val hasOwnKernels: Boolean get() = configured != null

    override val name: String get() = BackendNames.REFERENCE

    override val isPortable: Boolean get() = true

    override val unavailableReason: String? get() = null

    /** koblas's own implementation, so it runs anywhere koblas does. */
    override val isAvailable: Boolean get() = true

    override val priority: Int get() = 0

    /** This backend's kernels, or the process default when it was given none. */
    override val kernels: Kernels get() = configured ?: koblas.kernels
}

/** The shared portable backend, the fallback every seam resolves to when nothing else is registered. */
public val F64ReferenceBlas: ReferenceBackend = ReferenceBackend()
