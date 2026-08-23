package com.eignex.koblas.hostblas

import com.eignex.koblas.dense.host.cblas.OpenBlasConfig
import com.eignex.koblas.dense.host.jvm.HostBlasCalls

/** The CBLAS and LAPACKE halves resolved from one OpenBLAS and optional LAPACKE path pair. */
public class HostBackends(config: OpenBlasConfig = OpenBlasConfig()) {
    private val calls = HostBlasCalls(config)

    /** The CBLAS half. */
    public val blas: HostBlas = HostBlas(calls)

    /** The LAPACKE half. */
    public val lapack: HostLapack = HostLapack(calls, config)
}
