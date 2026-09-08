package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.backend.BackendNames

/**
 * [F64Blas] backed by the host's OpenBLAS through CBLAS, resolved with `dlopen` on first use.
 */
public class F64CblasBackend private constructor(private val blas: F64Cblas) :
    F64Blas by blas,
    F64RoutingBackend,
    BackendMetadataProvider {

    private constructor(loader: OpenBlasLoader, config: HostBlasConfig) : this(
        F64Cblas(requireNotNull(loader.cblas) { NO_OPENBLAS }, loader, config),
    )

    /** Resolves the CBLAS backend for a caller that wants to install it explicitly. */
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(OpenBlasLoader(config), config)

    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    override val isPortable: Boolean get() = false

    override val isAvailable: Boolean get() = blas.isAvailable

    override val unavailableReason: String? get() = blas.unavailableReason

    override val kernels: F64Kernels get() = blas.kernels

    override val backendMetadata: BackendMetadata get() = blas.backendMetadata

    override fun route(query: F64RouteQuery): BackendRoute? = blas.route(query)
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
