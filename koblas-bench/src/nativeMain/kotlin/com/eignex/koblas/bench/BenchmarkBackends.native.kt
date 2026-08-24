package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.host.cblas.F64CblasBackend
import com.eignex.koblas.dense.host.cblas.F64CblasKernels
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

/**
 * Returns the CBLAS backend explicitly because the linker may drop the unreferenced eager-init property,
 * leaving a silent reference-versus-reference run. Null when the host libraries are absent.
 */
internal actual fun nativeBackend(): F64LinearAlgebra? =
    if (F64CblasBackend.isAvailable()) F64CblasBackend() else null

/** Installs or clears the host CBLAS level-1 kernels, if the host has them at all. */
internal actual fun useHostLevel1(enabled: Boolean): Boolean {
    val kernels = if (enabled && F64CblasBackend.isAvailable()) F64CblasKernels() else null
    // The kernels are installed unrouted because the benchmark sweeps the lengths itself.
    installBackends(if (kernels == null) null else koblas.with(kernels = kernels))
    return kernels != null
}
