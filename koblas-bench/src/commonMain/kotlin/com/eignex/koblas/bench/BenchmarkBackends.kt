package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra

internal const val AUTO_BACKEND = "auto"

internal const val REFERENCE_BACKEND = "reference"

// Reaches the host library at every size. A run measuring a gate cannot obey it: the shipped value decides
// which side answers, so a curve taken under it is two half-curves rather than the two a crossover needs.
internal const val FORCED_BACKEND = "forced"

// Ungates the factorizations alone, leaving the level-1 to level-3 gates at their platform values. Forcing
// every gate at once measures more than one thing: a routine that stays portable still has its inner
// primitives replaced by host calls, so what looks like the host losing a solve is a portable algorithm
// paying a foreign call for each short dot and axpy inside it.
internal const val FORCED_FACTORIZE_BACKEND = "forced-factorize"

// The same for the gates a solve reads, so its crossover can be found without moving the factorization one.
internal const val FORCED_SOLVE_BACKEND = "forced-solve"

// Installs the host at its shipped gates, which is what the auto arm is for: it answers whether the value
// the library ships pays off, where the forced arm answers where the crossover actually is.
internal expect fun useShippedHost(): Boolean

internal expect fun useUngatedHost(): Boolean

internal expect fun useUngatedFactorization(): Boolean

// Ungates the level-2 and level-3 gates alone, which is what a solve over an existing factor reads. The
// kernel half is left alone for the same reason the factorize arm leaves it: replacing it would time a
// portable fallback against host primitives instead of the compiled-in ones.
internal expect fun useUngatedSolves(): Boolean

// The sparse LU gate counts stored entries, and the shipped value decides which side answers, so a sparse
// crossover needs the host ungated for the same reason the dense one does.
internal expect fun useUngatedSparseLu(): Boolean

/** Installs the sparse matrix half with its gate at zero, so a product of any size reaches the library. */
internal expect fun useUngatedSparseProduct(): Boolean

internal fun installBackend(backend: String) {
    // Cleared first, so an arm starts from the portable halves rather than whatever the previous one left.
    installBackends(null)
    // Each branch installs before reporting: an interpolation reads koblasInfo before the call beside it
    // runs, which would describe the state the arm was replacing.
    val installed = when (backend) {
        FORCED_BACKEND -> useUngatedHost()
        FORCED_FACTORIZE_BACKEND -> useUngatedFactorization()
        FORCED_SOLVE_BACKEND -> useUngatedSolves()
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
