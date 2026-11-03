package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.jvm25.simd.hardwareFusedMultiplyAdd
import com.eignex.koblas.internal.portable.PortableEngine
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The Vector API panels, and which of their bodies a given panel actually reaches: a panel whose shared vector
 * is strided is portable work, and so is one with fewer rows than a lane block unless it has a short body.
 *
 * Reached through the engine rather than by naming the object, because naming it resolves the species, which
 * on a runtime without the module is a linkage error before any test body runs.
 */
class SimdPanelKernelsTest {
    private val kernels: EngineKernels? = BuiltinEngines.simd?.panelProvider

    @Test
    fun `long column updates agree with the scalar oracle`() {
        val kernels = kernels ?: return skipped()

        assertLargeColumnUpdatesAgreeWithReference(kernels)
    }

    @Test
    fun `a panel with lanes to fill reaches the vector body and a shorter one does not`() {
        val kernels = kernels ?: return skipped()
        val lanes = lanes()

        for (work in vectorised()) {
            assertEquals(kernels.name, kernels.implementationFor(work, lanes, 4), "$work at $lanes rows")
            assertEquals(
                if (hasShortBody(work)) kernels.name else PortableEngine.name,
                kernels.implementationFor(work, lanes - 1, 4),
                "$work at ${lanes - 1} rows",
            )
        }
    }

    @Test
    fun `a short reduction or update runs a body of its own and is handed over whole`() {
        val kernels = kernels ?: return skipped()

        for (work in listOf(PanelWork.MultiDot, PanelWork.ColumnUpdate)) {
            for (rows in 1 until lanes()) {
                assertEquals(kernels.name, kernels.implementationFor(work, rows, 4096), "$work over $rows rows")
                assertEquals(
                    4096,
                    kernels.executionGroup(work, rows, 4096),
                    "$work over $rows rows is handed over whole",
                )
            }
        }
    }

    @Test
    fun `a strided shared vector is portable work at any width`() {
        val kernels = kernels ?: return skipped()

        for (work in vectorised()) {
            assertEquals(
                PortableEngine.name,
                kernels.implementationFor(work, WIDE, 4, contiguous = false),
                "$work strided",
            )
        }
    }

    // Vector work over adjacent right-hand sides and portable work over a strided group, which is the whole
    // of what staging one buys.
    @Test
    fun `the sparse right hand side panel names a vector body only where its sides are adjacent`() {
        val kernels = kernels ?: return skipped()
        val lanes = lanes()

        assertEquals(
            kernels.name,
            kernels.implementationFor(PanelWork.SparseRightHandSides, WIDE, 4),
            "adjacent right-hand sides",
        )
        assertEquals(
            PortableEngine.name,
            kernels.implementationFor(PanelWork.SparseRightHandSides, WIDE, 4, contiguous = false),
            "strided right-hand sides",
        )
        assertEquals(
            PortableEngine.name,
            kernels.implementationFor(PanelWork.SparseRightHandSides, lanes - 1, 4),
            "fewer adjacent right-hand sides than one lane block",
        )
        assertTrue(kernels.executionGroup(PanelWork.SparseRightHandSides, WIDE, 64) >= 1)
    }

    // A reduction reads the shared vector once for every column it is handed, so the grouping is the whole
    // column count rather than anything derived from the lane count.
    @Test
    fun `a contiguous reduction panel is handed every column at once`() {
        val kernels = kernels ?: return skipped()

        val group = kernels.executionGroup(PanelWork.MultiDot, WIDE, 64)

        assertEquals(64, group)
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
        PortableEngine.multiDot(1.0, a, 0, rows, x, 0, 1, rows, 1, 0.0, portable, 0, 1)

        val bound = rows * 2.220446049250313e-16 * portable[0]
        assertTrue(
            abs(vector[0] - portable[0]) <= bound,
            "fma=$hardwareFusedMultiplyAdd vector ${vector[0]} portable ${portable[0]}",
        )
    }

    /** A reduction or a column update shorter than a lane block has a body of its own. */
    private fun hasShortBody(work: PanelWork): Boolean = work == PanelWork.MultiDot || work == PanelWork.ColumnUpdate

    /** The species this backend resolved, read back from the name it publishes. */
    private fun lanes(): Int = checkNotNull(BuiltinEngines.simd).providers.first {
        EngineImplementation.Dot in it.implementations
    }.let { it as EngineKernels }.minimumLength(EngineImplementation.Dot)

    private fun vectorised() = PanelWork.entries

    private fun skipped() {
        println("SKIPPED: no Vector API module on this runtime; the panel bodies were not exercised")
    }

    private companion object {
        /** Wider than any lane block, so the vector body is reached wherever there is one. */
        const val WIDE = 512
    }
}
