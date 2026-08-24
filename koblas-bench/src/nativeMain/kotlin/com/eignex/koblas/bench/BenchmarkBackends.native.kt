package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.host.cblas.F64CblasBackend
import com.eignex.koblas.dense.host.cblas.F64CblasKernels
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

// Returned explicitly because the linker may drop the unreferenced eager-init property, leaving a silent
// reference-versus-reference run.
internal actual fun nativeBackend(): F64LinearAlgebra? =
    if (F64CblasBackend.isAvailable()) F64CblasBackend() else null

// Level 2, level 3 and the factorizations already dispatch to the host from size zero here, so only the
// level-1 gate needs overriding; the kernels are installed unrouted because the benchmark sweeps the
// lengths itself.
internal actual fun useUngatedHost(): Boolean {
    if (!F64CblasBackend.isAvailable()) return false
    val config = HostBlasConfig(level1Min = 0, level2Min = 0, level3Min = 0, factorizeMin = 0)
    val backend = F64CblasBackend(config)
    installBackends(koblas.with(kernels = F64CblasKernels(config), blas = backend, decompositions = backend))
    return true
}
