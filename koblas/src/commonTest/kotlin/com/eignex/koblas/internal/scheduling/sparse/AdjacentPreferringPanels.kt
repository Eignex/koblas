package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/**
 * A backend that asks for adjacent right-hand sides and computes with the portable bodies, so the staged
 * schedule runs on a platform with no vector bodies at all.
 *
 * Its group of three is deliberately not a lane count and not a power of two, so a call cuts groups the
 * shipped backends never would.
 */
internal object AdjacentPreferringPanels : KernelProvider by ScalarEngine {
    private const val GROUP = 3

    override val implementations = setOf(EngineImplementation.Panels)

    override val name: String get() = "adjacent-preferring"

    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int = when {
        !sparse(work) || !contiguous -> ScalarEngine.executionGroup(work, rows, columns, contiguous)
        columns <= 0 -> 1
        else -> minOf(GROUP, columns)
    }

    override fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean = sparse(work) && rows > 0

    /** Both sparse panel shapes, since what this stands in for is a backend that vectorises either. */
    private fun sparse(work: PanelWork): Boolean =
        work == PanelWork.SparseRightHandSides || work == PanelWork.SparseRightHandSideReduction
}

/**
 * Every sparse engine this runtime has, so a rule is checked where it is implemented and where selected:
 * the portable panels, the staged schedule over [AdjacentPreferringPanels], the platform default, and the
 * vector engine where there is one.
 */
internal fun sparsePanelEngines(): List<Pair<String, KoblasEngine>> = buildList {
    add("strided" to engineWithKernels(panels = ScalarEngine))
    add("staged" to engineWithKernels(panels = AdjacentPreferringPanels))
    add("selected" to koblas)
    KoblasEngine.simd?.let { add("simd" to it) }
}
