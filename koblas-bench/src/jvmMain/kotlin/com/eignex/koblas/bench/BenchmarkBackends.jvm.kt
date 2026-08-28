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
// back and every half is installed explicitly.
internal actual fun useHost(): Boolean = installHost(HostBlasConfig())

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

// Whichever sparse library this JVM can load. KLU first, matching backend registration priority.
internal actual fun useSparseProduct(): Boolean {
    val cholmod = CholmodSparseBlas()
    if (!cholmod.isAvailable) return false
    installBackends(koblas.with(sparseBlas = cholmod))
    return true
}

internal actual fun useSparseLu(): Boolean {
    val backends = F64SparseBackends(
        KluConfig(),
        UmfpackConfig(),
    )
    val chosen = listOf(backends.klu, backends.umfpack).firstOrNull { it.isAvailable } ?: return false
    installBackends(koblas.with(sparseDecompositions = chosen))
    return true
}
