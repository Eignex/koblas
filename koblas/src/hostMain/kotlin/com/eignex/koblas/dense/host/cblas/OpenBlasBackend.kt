package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.backend.BackendNames

/**
 * [Blas] backed by the host's OpenBLAS through CBLAS, resolved with `dlopen` on first use.
 */
public class OpenBlasBackend private constructor(private val blas: OpenBlas) :
    Blas by blas,
    RoutingBackend,
    BackendMetadataProvider {

    private constructor(loader: OpenBlasLoader, config: HostBlasConfig) : this(
        OpenBlas(requireNotNull(loader.cblas) { NO_OPENBLAS }, loader, config),
    )

    /** Resolves the CBLAS backend for a caller that wants to install it explicitly. */
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(OpenBlasLoader(config), config)

    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    override val isPortable: Boolean get() = false

    override val isAvailable: Boolean get() = blas.isAvailable

    override val unavailableReason: String? get() = blas.unavailableReason

    override val kernels: Kernels get() = blas.kernels

    override val backendMetadata: BackendMetadata get() = blas.backendMetadata

    override fun route(query: RouteQuery): BackendRoute? = blas.route(query)
        ?.let { if (it.execution == BackendExecution.NATIVE) it.copy(executor = name) else it }

    /** Availability checks for the host CBLAS. */
    public companion object {
        /** Whether the host provides the CBLAS subset koblas binds. */
        public fun isAvailable(config: HostBlasConfig = HostBlasConfig()): Boolean {
            val loader = OpenBlasLoader(config)
            return loader.cblas != null
        }

        private const val NO_OPENBLAS =
            "OpenBLAS is not available on this host; koblas falls back to the reference backend"
    }
}
