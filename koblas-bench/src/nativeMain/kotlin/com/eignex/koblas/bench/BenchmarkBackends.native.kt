package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.host.cblas.F64CblasBackend
import com.eignex.koblas.dense.host.cblas.F64CblasKernels
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

internal actual fun nativeBackend(): F64LinearAlgebra? =
    if (F64CblasBackend.isAvailable()) F64CblasBackend() else null

internal actual fun useHostLevel1(enabled: Boolean): Boolean {
    val kernels = if (enabled && F64CblasBackend.isAvailable()) F64CblasKernels() else null
    installBackends(if (kernels == null) null else koblas.with(kernels = kernels))
    return kernels != null
}
