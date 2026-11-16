package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.BackendEngine

/** Primitive arithmetic and provenance implemented by a backend, independent of the typed public API. */
internal interface KernelProvider {
    val name: String
    val id: String get() = vendor?.vendorName?.lowercase() ?: name
    val implementations: Set<EngineImplementation> get() = emptySet()
    val vendor: Vendor? get() = null
    val libraryPath: String get() = ""
    val version: String get() = ""
    val threadEvidence: ThreadEvidence get() = ThreadEvidence.Unconfirmed
    val providers: List<KernelProvider> get() = listOf(this)

    /** Whether attribution can depend on the values processed by this contribution. */
    fun dynamicAttribution(implementation: EngineImplementation): Boolean = false

    /** Whether this contribution requires unit strides. */
    fun requiresUnitStride(implementation: EngineImplementation): Boolean = false

    /** Smallest run the contribution can execute. The built-in engine checks it before calling the provider. */
    fun minimumLength(implementation: EngineImplementation): Int = 0
}

internal fun KoblasEngine.unsupported(): Nothing =
    throw UnsupportedOperationException("This operation is not supplied by $name")

/** The public adapter is created at engine selection, outside numerical execution. */
internal val KoblasEngine.kernels: KernelProvider get() = (this as? BackendEngine)?.backend ?: unsupported()

/** Typed vector seams are resolved outside arithmetic loops. */
internal val KernelProvider.vectorKernels: VectorKernels get() = this as? VectorKernels ?: missingRole()
internal val KernelProvider.indexedKernels: IndexedKernels get() = this as? IndexedKernels ?: missingRole()

internal fun KernelProvider.missingRole(): Nothing =
    throw UnsupportedOperationException("This operation is not supplied by $name")
