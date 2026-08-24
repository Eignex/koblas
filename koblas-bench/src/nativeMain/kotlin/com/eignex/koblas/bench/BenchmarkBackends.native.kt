package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.host.cblas.F64CblasLinearAlgebra
import com.eignex.koblas.dense.host.cblas.F64CblasVectorKernels
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas

// An explicit reference keeps the linker from dropping eager backend initialization.
internal actual fun nativeBackend(): F64LinearAlgebra? =
    if (F64CblasLinearAlgebra.isAvailable()) F64CblasLinearAlgebra() else null

internal actual fun useHostLevel1(enabled: Boolean): Boolean {
    val kernels = if (enabled && F64CblasLinearAlgebra.isAvailable()) F64CblasVectorKernels() else null
    // The kernels are installed unrouted because the benchmark sweeps the lengths itself.
    installBackends(if (kernels == null) null else koblas.with(vectorKernels = kernels))
    return kernels != null
}
