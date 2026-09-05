package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.dense.host.F64KernelsAdapter
import com.eignex.koblas.internal.backend.BackendNames

/**
 * The host OpenBLAS behind koblas's level-1 primitives, bound with cinterop. Every routine lives in
 * [F64KernelsAdapter]; this supplies the native entry points and the backend's identity.
 */
public class F64CblasKernels internal constructor(
    private val loader: OpenBlasLoader,
    /** Policy for this level-1 backend instance. */
    public val config: HostBlasConfig,
) : F64KernelsAdapter(NativeCblasKernelCalls(loader)) {
    public constructor(config: HostBlasConfig = HostBlasConfig()) : this(OpenBlasLoader(config), config)

    override val name: String get() = BackendNames.CBLAS

    /** The level-1 kernels come from CBLAS. */
    override val isAvailable: Boolean get() = loader.cblas != null
}
