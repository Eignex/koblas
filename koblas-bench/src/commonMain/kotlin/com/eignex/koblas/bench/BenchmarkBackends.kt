package com.eignex.koblas.bench

import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo

/** The platform's native backend, or null to leave resolution to discovery. */
internal expect fun nativeBackend(): LinearAlgebra?

/**
 * Turns the platform's host level-1 kernels on or off, reporting whether any are now installed. They sit
 * below the [LinearAlgebra] seam, so the backend parameter does not reach them.
 */
internal expect fun useHostLevel1(enabled: Boolean): Boolean

/**
 * Installs the named backend and prints what resolved. [REFERENCE_BACKEND] forces the portable kernels,
 * anything else takes the platform's native backend if it has one.
 */
internal fun installBackend(backend: String) {
    val chosen = if (backend == REFERENCE_BACKEND) ReferenceLinearAlgebra else nativeBackend()
    installBackends(chosen?.let { koblas.with(blas = it, lapack = it) })
    println("resolved: $koblasInfo")
}
