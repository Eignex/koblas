package com.eignex.koblas.bench

import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.jvm.F64Backends
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.cholmod.CholmodSparseBlas
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.F64Context
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

// The CBLAS and LAPACKE halves are separate objects here, so there is no single F64LinearAlgebra to hand
// back and every half is installed explicitly. Leaving it to discovery is what made the auto arm run the
// portable path against itself.
internal actual fun useShippedHost(): Boolean = installHost(HostBlasConfig())

// Gated from zero because the shipped gate is what a run measuring it must not obey: with the Vector API
// kernels in place level1 and level2 are Int.MAX_VALUE, so an arm left to the default routes every call back
// to the compiled-in path and times that twice. Installed unrouted for the reason the native binding gives,
// that the benchmark sweeps the sizes itself.
/**
 * Only the factorization gates move. The kernel half is deliberately left alone: [F64Context.with] installs
 * whatever it is handed without the routing wrapper, so passing the host kernels here would put every short
 * dot and axpy through a foreign call whatever level1Min says, and a routine that stayed portable would be
 * timed against that instead of against the compiled-in kernels.
 */
internal actual fun useUngatedSolves(): Boolean = installHalves(HostBlasConfig(level2Min = 0, level3Min = 0))

internal actual fun useUngatedFactorization(): Boolean {
    return installHalves(HostBlasConfig(factorizeMin = 0))
}

/** Both halves a decomposition needs, and not the kernels, for the reason above. */
private fun installHalves(config: HostBlasConfig): Boolean {
    val backends = F64Backends(config)
    if (!backends.blas.isAvailable) return false
    installBackends(koblas.with(blas = backends.blas, decompositions = backends.decompositions))
    return true
}

internal actual fun useUngatedHost(): Boolean = installHost(
    HostBlasConfig(level1Min = 0, level2Min = 0, level3Min = 0, factorizeMin = 0),
)

private fun installHost(config: HostBlasConfig): Boolean {
    val backends = F64Backends(config)
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
internal actual fun useUngatedSparseProduct(): Boolean {
    val cholmod = CholmodSparseBlas(level2Min = 0)
    if (!cholmod.isAvailable) return false
    installBackends(koblas.with(sparseBlas = cholmod))
    return true
}

internal actual fun useUngatedSparseLu(): Boolean {
    val backends = F64SparseBackends(KluConfig(factorizeMin = 0), UmfpackConfig(factorizeMin = 0))
    val chosen = listOf(backends.klu, backends.umfpack).firstOrNull { it.isAvailable } ?: return false
    installBackends(koblas.with(sparseDecompositions = chosen))
    return true
}
