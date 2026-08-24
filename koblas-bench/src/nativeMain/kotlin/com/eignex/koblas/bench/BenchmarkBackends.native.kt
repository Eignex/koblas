package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.host.cblas.F64CblasLinearAlgebra
import com.eignex.koblas.dense.host.cblas.F64CblasVectorKernels
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

internal actual fun nativeBackend(): F64LinearAlgebra? =
    if (F64CblasLinearAlgebra.isAvailable()) F64CblasLinearAlgebra() else null

internal actual fun useHostLevel1(enabled: Boolean): Boolean {
    val kernels = if (enabled && F64CblasLinearAlgebra.isAvailable()) F64CblasVectorKernels() else null
    installBackends(if (kernels == null) null else koblas.with(vectorKernels = kernels))
    return kernels != null
}
