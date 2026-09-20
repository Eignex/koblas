package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.internal.numeric.hardwareFusedMultiplyAdd
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * The Vector API panels, and which of their bodies a given panel actually reaches.
 *
 * The arithmetic is checked against the written-out definitions rather than against the portable backend, so
 * a body that agreed with the portable one only by delegating to it would still have to be right. What is
 * particular to this backend is where it stops: a panel with fewer rows than a lane block, or one whose
 * shared vector is strided, is portable work, and saying so is the difference between an attribution and a
 * label.
 *
 * Reached through the engine rather than by naming the object, because naming it resolves the species: on a
 * runtime without the module that is a linkage error before any test body runs, and that absence is exactly
 * one of the configurations this has to survive.
 */
class SimdPanelKernelsTest {
    private val kernels: DensePanelKernels? = BuiltinEngines.simd?.panelKernels

    @Test
    fun `the vector panels agree with the written out definitions`() {
        val kernels = kernels ?: return skipped()

        assertPanelKernelsAgreeWithReference(kernels)
    }

    @Test
    fun `the vector panels keep the no read and zero evaluation rules`() {
        val kernels = kernels ?: return skipped()

        assertPanelContractHolds(kernels)
    }

    @Test
    fun `a panel with lanes to fill reaches the vector body and a shorter one does not`() {
        val kernels = kernels ?: return skipped()
        val lanes = lanes(kernels)

        for (work in vectorised()) {
            assertEquals(kernels.name, kernels.implementationFor(work, lanes, 4), "$work at $lanes rows")
            assertEquals(
                PortablePanelKernels.name,
                kernels.implementationFor(work, lanes - 1, 4),
                "$work at ${lanes - 1} rows",
            )
        }
    }

    @Test
    fun `a strided shared vector is portable work at any width`() {
        val kernels = kernels ?: return skipped()

        for (work in vectorised()) {
            assertEquals(
                PortablePanelKernels.name,
                kernels.implementationFor(work, WIDE, 4, contiguous = false),
                "$work strided",
            )
        }
    }

    /**
     * The sparse right-hand-side panel is vector work over adjacent right-hand sides and portable work over
     * a strided group, which is the whole of what staging one buys and the reason a route asks about the
     * layout rather than about the engine.
     */
    @Test
    fun `the sparse right hand side panel names a vector body only where its sides are adjacent`() {
        val kernels = kernels ?: return skipped()
        val lanes = lanes(kernels)

        assertEquals(
            kernels.name,
            kernels.implementationFor(PanelWork.SparseRightHandSides, WIDE, 4),
            "adjacent right-hand sides",
        )
        assertEquals(
            PortablePanelKernels.name,
            kernels.implementationFor(PanelWork.SparseRightHandSides, WIDE, 4, contiguous = false),
            "strided right-hand sides",
        )
        assertEquals(
            PortablePanelKernels.name,
            kernels.implementationFor(PanelWork.SparseRightHandSides, lanes - 1, 4),
            "fewer adjacent right-hand sides than one lane block",
        )
        assertTrue(kernels.executionGroup(PanelWork.SparseRightHandSides, WIDE, 64) >= 1)
    }

    /**
     * The grouping is the backend's own choice and is not the lane count, which is the confusion the whole
     * seam exists to prevent. They agree here only where the measurement happened to land on the same number.
     */
    @Test
    fun `the recommended grouping is a column count rather than a lane count`() {
        val kernels = kernels ?: return skipped()

        val shared = kernels.executionGroup(PanelWork.MultiDot, WIDE, 64)
        val separate = kernels.executionGroup(PanelWork.RankUpdate, WIDE, 64)

        assertNotEquals(shared, separate, "one grouping served both panel shapes")
        assertTrue(shared in 1..64 && separate in 1..64)
    }

    /**
     * A fused multiply-add rounds once where a multiply and an add round twice, so a machine with the
     * instruction can differ from the portable body in the last place and must not differ by more.
     */
    @Test
    fun `the vector body agrees with the portable one to within one rounding of each term`() {
        val kernels = kernels ?: return skipped()
        val rows = 1000
        val a = DoubleArray(rows) { 1.0 / (it + 1) }
        val x = DoubleArray(rows) { 1.0 / (it + 3) }

        val vector = DoubleArray(1)
        kernels.multiDot(1.0, a, 0, rows, x, 0, 1, rows, 1, 0.0, vector, 0, 1)
        val portable = DoubleArray(1)
        PortablePanelKernels.multiDot(1.0, a, 0, rows, x, 0, 1, rows, 1, 0.0, portable, 0, 1)

        val bound = rows * 2.220446049250313e-16 * portable[0]
        assertTrue(
            kotlin.math.abs(vector[0] - portable[0]) <= bound,
            "fma=$hardwareFusedMultiplyAdd vector ${vector[0]} portable ${portable[0]}",
        )
    }

    /** The species this backend resolved, read back from the name it publishes. */
    private fun lanes(kernels: DensePanelKernels): Int = kernels.name.substringAfter('(').substringBefore(' ').toInt()

    private fun vectorised() = PanelWork.entries

    private fun skipped() {
        println("SKIPPED: no Vector API module on this runtime; the panel bodies were not exercised")
    }

    private companion object {
        /** Wider than any lane block, so the vector body is reached wherever there is one. */
        const val WIDE = 512
    }
}
