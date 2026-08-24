package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.jvm.HostBlasCalls

/** The CBLAS and LAPACKE halves resolved from one OpenBLAS and optional LAPACKE path pair. */
public class F64Backends(config: HostBlasConfig = HostBlasConfig()) {
    private val calls = HostBlasCalls(config)

    /** The CBLAS half. */
    public val blas: F64Cblas = F64Cblas(calls)

    /** The level-1 CBLAS half. */
    public val kernels: F64CblasKernels = F64CblasKernels(calls, config)

    /** The LAPACKE half. */
    public val lapack: F64Lapacke = F64Lapacke(calls, config)
}
