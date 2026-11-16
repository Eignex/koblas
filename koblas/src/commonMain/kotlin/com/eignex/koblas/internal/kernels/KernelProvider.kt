package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.BackendEngine

/** Primitive arithmetic and provenance implemented by a backend, independent of the typed public API. */
internal interface KernelProvider : EngineKernels {
    val id: String get() = vendor?.vendorName?.lowercase() ?: name
    val implementations: Set<EngineImplementation> get() = emptySet()
    val vendor: Vendor? get() = null
    val libraryPath: String get() = ""
    val version: String get() = ""
    val threadEvidence: ThreadEvidence get() = ThreadEvidence.Unconfirmed
    val providers: List<KernelProvider> get() = listOf(this)
}

internal fun KoblasEngine.unsupported(): Nothing =
    throw UnsupportedOperationException("This operation is not supplied by $name")

/** The public adapter is created at engine selection, outside numerical execution. */
internal val KoblasEngine.kernels: KernelProvider get() = (this as? BackendEngine)?.backend ?: unsupported()
