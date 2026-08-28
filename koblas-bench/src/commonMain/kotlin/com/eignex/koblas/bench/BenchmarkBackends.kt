package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra

internal const val REFERENCE_BACKEND = "reference"
internal const val HOST_BACKEND = "host"

internal expect fun useHost(): Boolean

internal expect fun useSparseLu(): Boolean

internal expect fun useSparseProduct(): Boolean

internal fun installBackend(backend: String) {
    // Cleared first, so an arm starts from the portable halves rather than whatever the previous one left.
    installBackends(null)
    // Each branch installs before reporting: an interpolation reads koblasInfo before the call beside it
    // runs, which would describe the state the arm was replacing.
    val installed = when (backend) {
        HOST_BACKEND -> useHost()
        else -> {
            installBackends(
                koblas.with(blas = F64ReferenceLinearAlgebra, decompositions = F64ReferenceLinearAlgebra),
            )
            true
        }
    }
    println("resolved: $koblasInfo (installed=$installed)")
}
