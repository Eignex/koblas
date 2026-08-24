package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo

internal const val AUTO_BACKEND = "auto"

internal const val REFERENCE_BACKEND = "reference"

internal const val BUILTIN_KERNELS = "builtin"

internal const val HOST_KERNELS = "host"

internal expect fun nativeBackend(): F64LinearAlgebra?

internal expect fun useHostLevel1(enabled: Boolean): Boolean

internal fun installBackend(backend: String) {
    val chosen = if (backend == REFERENCE_BACKEND) F64ReferenceLinearAlgebra else nativeBackend()
    installBackends(chosen?.let { koblas.with(blas = it, lapack = it) })
    println("resolved: $koblasInfo")
}
