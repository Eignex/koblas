package com.eignex.koblas.bench

import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo

internal const val AUTO_BACKEND = "auto"

internal const val REFERENCE_BACKEND = "reference"

// Reaches the host library at every size. A run measuring a gate cannot obey it: the shipped value decides
// which side answers, so a curve taken under it is two half-curves rather than the two a crossover needs.
internal const val FORCED_BACKEND = "forced"

internal expect fun nativeBackend(): F64LinearAlgebra?

internal expect fun useUngatedHost(): Boolean

internal fun installBackend(backend: String) {
    // Cleared first, so an arm inherits discovery rather than whatever the previous one installed.
    installBackends(null)
    if (backend == FORCED_BACKEND) {
        println("resolved: $koblasInfo (ungated=${useUngatedHost()})")
        return
    }
    val chosen = if (backend == REFERENCE_BACKEND) F64ReferenceLinearAlgebra else nativeBackend()
    installBackends(chosen?.let { koblas.with(blas = it, decompositions = it) })
    println("resolved: $koblasInfo")
}
