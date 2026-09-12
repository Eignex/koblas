package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.NativeContext
import com.eignex.koblas.internal.kernels.NativeHost
import com.eignex.koblas.internal.kernels.NativeKernel
import com.eignex.koblas.internal.kernels.nativeEligibility

/** Runtime competitors have different host-call costs even when they share a native candidate. */
internal enum class RuntimeCompetitor { JvmScalar, JvmVector, Native }

/**
 * One operation-compatible candidate, prebound from the catalog by exact ID. The profile chooses preference;
 * the probe supplies legality. Current context is obtained only after a host-call rule allows native work.
 */
internal class NativeSelection(
    val kernel: NativeKernel?,
    private val host: NativeHost?,
    val crossovers: HostCrossovers,
    private val currentContext: () -> NativeContext,
) {
    fun auto(runtime: RuntimeCompetitor, rows: Int, columns: Int = 1, depth: Int = 1): NativeKernel? {
        require(rows >= 0 && columns >= 0 && depth >= 0) { "negative operation shape" }
        val rule = when (runtime) {
            RuntimeCompetitor.JvmScalar -> crossovers.scalarToC
            RuntimeCompetitor.JvmVector -> crossovers.simdToC
            RuntimeCompetitor.Native -> crossovers.nativeToC
        }
        if (!rule.accepts(rows, columns, depth)) return null
        val candidate = kernel ?: return null
        val facts = host ?: return null
        return candidate.takeIf { nativeEligibility(it, facts, currentContext()) == 0 }
    }

    fun exact(): NativeKernel {
        val candidate = requireNotNull(kernel) { "exact native implementation is not built" }
        val facts = requireNotNull(host) { "native catalog is unavailable" }
        val reason = nativeEligibility(candidate, facts, currentContext())
        require(reason == 0) { "exact native implementation ${candidate.id} unavailable: reason=$reason" }
        return candidate
    }
}
