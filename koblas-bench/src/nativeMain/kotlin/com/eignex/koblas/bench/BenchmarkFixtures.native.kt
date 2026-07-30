package com.eignex.koblas.bench

import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.cblas.CblasLinearAlgebra

/**
 * The host's CBLAS/LAPACKE backend when its libraries are present, otherwise `null`.
 *
 * A `null` here means an `auto` run measures the portable kernels, which the printed resolved line makes
 * visible — the alternative, an unnoticed reference-versus-reference comparison, is the failure mode this
 * explicit install exists to prevent.
 */
internal actual fun nativeBackend(): LinearAlgebra? =
    if (CblasLinearAlgebra.isAvailable()) CblasLinearAlgebra() else null
