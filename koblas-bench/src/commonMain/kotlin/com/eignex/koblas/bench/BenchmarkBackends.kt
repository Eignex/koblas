package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra

internal const val AUTO_BACKEND = "auto"

internal const val REFERENCE_BACKEND = "reference"

// Reaches the host library at every size. A run measuring a gate cannot obey it: the shipped value decides
// which side answers, so a curve taken under it is two half-curves rather than the two a crossover needs.
internal const val FORCED_BACKEND = "forced"

// Installs the host at its shipped gates, which is what the auto arm is for: it answers whether the value
// the library ships pays off, where the forced arm answers where the crossover actually is.
internal expect fun useShippedHost(): Boolean

internal expect fun useUngatedHost(): Boolean

// The sparse LU gate counts stored entries, and the shipped value decides which side answers, so a sparse
// crossover needs the host ungated for the same reason the dense one does.
internal expect fun useUngatedSparseLu(): Boolean

internal fun installBackend(backend: String) {
    // Cleared first, so an arm starts from the portable halves rather than whatever the previous one left.
    installBackends(null)
    // Each branch installs before reporting: an interpolation reads koblasInfo before the call beside it
    // runs, which would describe the state the arm was replacing.
    val installed = when (backend) {
        FORCED_BACKEND -> useUngatedHost()
        AUTO_BACKEND -> useShippedHost()
        else -> {
            installBackends(
                koblas.with(blas = F64ReferenceLinearAlgebra, decompositions = F64ReferenceLinearAlgebra),
            )
            true
        }
    }
    println("resolved: $koblasInfo (installed=$installed)")
}
