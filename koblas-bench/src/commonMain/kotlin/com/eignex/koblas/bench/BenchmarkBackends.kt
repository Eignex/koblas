package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo

/** The backend parameter value that leaves resolution to discovery. */
internal const val AUTO_BACKEND = "auto"

/** The backend parameter value that pins a run to the portable kernels. */
internal const val REFERENCE_BACKEND = "reference"

/** The kernels parameter value that clears the platform's level-1 kernels. */
internal const val BUILTIN_KERNELS = "builtin"

/** The kernels parameter value that installs them. */
internal const val HOST_KERNELS = "host"

/** The platform's native backend, or null to leave resolution to discovery. */
internal expect fun nativeBackend(): F64LinearAlgebra?

/**
 * Turns the platform's host level-1 kernels on or off, reporting whether any are now installed. They sit
 * below the [F64LinearAlgebra] seam, so the backend parameter does not reach them.
 */
internal expect fun useHostLevel1(enabled: Boolean): Boolean

/**
 * Installs the named backend and prints what resolved. [REFERENCE_BACKEND] forces the portable kernels,
 * anything else takes the platform's native backend if it has one.
 */
internal fun installBackend(backend: String) {
    val chosen = if (backend == REFERENCE_BACKEND) F64ReferenceLinearAlgebra else nativeBackend()
    installBackends(chosen?.let { koblas.with(blas = it, lapack = it) })
    println("resolved: $koblasInfo")
}
