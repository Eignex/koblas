package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.backend.BackendNames

/**
 * [F64LinearAlgebra] backed by the host's OpenBLAS through CBLAS and LAPACKE, resolved with `dlopen` on
 * first use. The two halves resolve independently, so [isAvailable] reports whether both did.
 */
public class F64CblasBackend private constructor(private val blas: F64Cblas, private val decompositions: F64Lapacke) :
    F64LinearAlgebra,
    F64Blas by blas,
    F64Decompositions by decompositions,
    F64RoutingBackend,
    BackendMetadataProvider {

    private constructor(loader: OpenBlasLoader, config: HostBlasConfig) : this(
        F64Cblas(requireNotNull(loader.cblas) { NO_OPENBLAS }, loader, config),
        F64Lapacke(requireNotNull(loader.lapacke) { NO_LAPACKE }, requireNotNull(loader.cblas), loader, config),
    )

    /** Resolves both halves, for a caller that wants to install the backend explicitly. */
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(OpenBlasLoader(config), config)

    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    override val isPortable: Boolean get() = false

    /** Both halves, since this type is the pair. Either half alone is reachable through the registry. */
    override val isAvailable: Boolean get() = blas.isAvailable && decompositions.isAvailable

    /** The BLAS half's kernels, so both halves' inherited routines agree. */
    override val kernels: F64Kernels get() = blas.kernels

    override val backendMetadata: BackendMetadata get() = blas.backendMetadata

    override fun route(query: F64RouteQuery): BackendRoute? = when (query.role) {
        BackendRole.DENSE_BLAS -> blas.route(query)
        BackendRole.DENSE_DECOMPOSITIONS -> decompositions.route(query)
        else -> null
    }?.let { if (it.execution == BackendExecution.NATIVE) it.copy(executor = name) else it }

    /** Availability checks for the host CBLAS and LAPACKE. */
    public companion object {
        /** Whether the host provides both CBLAS and LAPACKE, so the full backend can be constructed. */
        public fun isAvailable(config: HostBlasConfig = HostBlasConfig()): Boolean {
            val loader = OpenBlasLoader(config)
            return loader.cblas != null && loader.lapacke != null
        }

        private const val NO_OPENBLAS =
            "OpenBLAS is not available on this host; koblas falls back to the reference backend"
        private const val NO_LAPACKE =
            "LAPACKE is not available on this host; koblas keeps its portable factorizations"
    }
}
