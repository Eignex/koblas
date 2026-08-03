package com.eignex.koblas.bench

import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.cblas.CblasLevel1Kernels
import com.eignex.koblas.cblas.CblasLinearAlgebra
import com.eignex.koblas.installLevel1

/**
 * The host's CBLAS/LAPACKE backend when its libraries are present, otherwise `null`.
 *
 * A `null` here means an `auto` run measures the portable kernels, which the printed resolved line makes
 * visible — the alternative, an unnoticed reference-versus-reference comparison, is the failure mode this
 * explicit install exists to prevent.
 */
internal actual fun nativeBackend(): LinearAlgebra? =
    if (CblasLinearAlgebra.isAvailable()) CblasLinearAlgebra() else null

/** Installs or clears the host CBLAS level-1 kernels, if the host has them at all. */
internal actual fun useHostLevel1(enabled: Boolean): Boolean {
    val kernels = if (enabled && CblasLinearAlgebra.isAvailable()) CblasLevel1Kernels() else null
    installLevel1(kernels)
    return kernels != null
}
