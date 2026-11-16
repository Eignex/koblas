package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.KernelProvider
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/** One shared typed entry layer for every primitive backend. */
internal class BackendEngine(internal val backend: KernelProvider) : KoblasEngine {
    override val name: String get() = backend.name
    override val id: String get() = backend.id
    override val implementations: Set<EngineImplementation> get() = backend.implementations
    override val vendor: Vendor? get() = backend.vendor
    override val libraryPath: String get() = backend.libraryPath
    override val version: String get() = backend.version
    override val threadEvidence: ThreadEvidence get() = backend.threadEvidence
    override val providers: List<KoblasEngine> = backend.providers.map {
        if (it === backend) this else BackendEngine(it)
    }
}

/** Construction belongs to selection or startup, never to a numerical schedule. */
internal fun KernelProvider.asEngine(): KoblasEngine =
    if (this === ScalarEngine) scalarPublicEngine else BackendEngine(this)

internal val scalarPublicEngine: KoblasEngine = BackendEngine(ScalarEngine)
internal val simdPublicEngine: KoblasEngine? by lazy { simdEngine?.asEngine() }
