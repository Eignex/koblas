package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.jvm.F64Backends
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

internal actual fun nativeBackend(): F64LinearAlgebra? = null

// Gated from zero because the shipped gate is what a run measuring it must not obey: with the Vector API
// kernels in place level1 and level2 are Int.MAX_VALUE, so an arm left to the default routes every call back
// to the compiled-in path and times that twice. Installed unrouted for the reason the native binding gives,
// that the benchmark sweeps the sizes itself.
internal actual fun useUngatedHost(): Boolean {
    val backends = F64Backends(
        HostBlasConfig(level1Min = 0, level2Min = 0, level3Min = 0, factorizeMin = 0, factorizeRhsMin = 1),
    )
    if (!backends.blas.isAvailable) return false
    installBackends(
        koblas.with(
            kernels = backends.kernels,
            blas = backends.blas,
            decompositions = backends.decompositions,
        ),
    )
    return true
}

// Whichever sparse library this JVM can load, ungated. KLU first, matching the priority the backends
// register with, so the arm measures the half discovery would have picked.
internal actual fun useUngatedSparseLu(): Boolean {
    val backends = F64SparseBackends(KluConfig(factorizeMin = 0), UmfpackConfig(factorizeMin = 0))
    val chosen = listOf(backends.klu, backends.umfpack).firstOrNull { it.isAvailable } ?: return false
    installBackends(koblas.with(sparseLu = chosen))
    return true
}
