package com.eignex.koblas.bench

import com.eignex.koblas.cblas.CblasLinearAlgebra
import com.eignex.koblas.cblas.CblasVectorKernels
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

/**
 * Returns the CBLAS backend explicitly because the linker may drop the unreferenced eager-init property,
 * leaving a silent reference-versus-reference run. Null when the host libraries are absent.
 */
internal actual fun nativeBackend(): LinearAlgebra? =
    if (CblasLinearAlgebra.isAvailable()) CblasLinearAlgebra() else null

/** Installs or clears the host CBLAS level-1 kernels, if the host has them at all. */
internal actual fun useHostLevel1(enabled: Boolean): Boolean {
    val kernels = if (enabled && CblasLinearAlgebra.isAvailable()) CblasVectorKernels() else null
    // The kernels are installed unrouted because the benchmark sweeps the lengths itself.
    installBackends(if (kernels == null) null else koblas.with(vectorKernels = kernels))
    return kernels != null
}
