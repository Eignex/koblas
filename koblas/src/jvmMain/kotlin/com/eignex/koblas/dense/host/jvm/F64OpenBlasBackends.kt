package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.cblas.OpenBlasConfig
import com.eignex.koblas.dense.host.jvm.HostBlasCalls

/** The CBLAS and LAPACKE halves resolved from one OpenBLAS and optional LAPACKE path pair. */
public class F64OpenBlasBackends(config: OpenBlasConfig = OpenBlasConfig()) {
    private val calls = HostBlasCalls(config)

    /** The CBLAS half. */
    public val blas: F64OpenBlasBlas = F64OpenBlasBlas(calls)

    /** The level-1 CBLAS half. */
    public val vectorKernels: F64OpenBlasVectorKernels = F64OpenBlasVectorKernels(calls, config)

    /** The LAPACKE half. */
    public val lapack: F64OpenBlasLapack = F64OpenBlasLapack(calls, config)
}
