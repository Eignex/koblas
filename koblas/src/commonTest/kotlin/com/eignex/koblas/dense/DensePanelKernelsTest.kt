package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The panel contract, on the portable backend and on whichever one this platform selected.
 *
 * The two are checked against the same written-out definitions rather than against each other, so a backend
 * that agrees with the portable one only because it delegates to it still has to be right.
 */
class DensePanelKernelsTest {
    @Test
    fun `the portable panels agree with the written out definitions`() {
        assertPanelKernelsAgreeWithReference(PortablePanelKernels)
    }

    @Test
    fun `the selected panels agree with the written out definitions`() {
        assertPanelKernelsAgreeWithReference(koblas.panelKernels)
    }

    @Test
    fun `the candidate panels agree with the written out definitions`() {
        val candidate = BuiltinEngines.simd ?: return skipped()
        assertPanelKernelsAgreeWithReference(candidate.panelKernels)
    }

    @Test
    fun `the portable panels keep the no read and zero evaluation rules`() {
        assertPanelContractHolds(PortablePanelKernels)
    }

    @Test
    fun `the selected panels keep the no read and zero evaluation rules`() {
        assertPanelContractHolds(koblas.panelKernels)
    }

    @Test
    fun `the candidate panels keep the no read and zero evaluation rules`() {
        val candidate = BuiltinEngines.simd ?: return skipped()
        assertPanelContractHolds(candidate.panelKernels)
        assertEmptyExtentsReadNothing(candidate.panelKernels)
        assertPanelsStayInsideTheirWindows(candidate.panelKernels)
    }

    @Test
    fun `the portable panels do nothing for an empty extent`() {
        assertEmptyExtentsReadNothing(PortablePanelKernels)
        assertPanelsStayInsideTheirWindows(PortablePanelKernels)
    }

    @Test
    fun `the selected panels do nothing for an empty extent`() {
        assertEmptyExtentsReadNothing(koblas.panelKernels)
        assertPanelsStayInsideTheirWindows(koblas.panelKernels)
    }

    @Test
    fun `every backend recommends a usable grouping`() {
        assertExecutionGroupIsUsable(PortablePanelKernels)
        assertExecutionGroupIsUsable(koblas.panelKernels)
        BuiltinEngines.scalar.panelKernels.let(::assertExecutionGroupIsUsable)
        BuiltinEngines.simd?.panelKernels?.let(::assertExecutionGroupIsUsable)
        BuiltinEngines.simd?.panelKernels?.let(::assertExecutionGroupIsUsable)
    }

    /**
     * A logical panel is the same arithmetic however it is split, which is what lets a caller use a grouping
     * other than the one it was given.
     *
     * Splitting a multi-dot writes each output once whatever the split, and splitting a column update or a
     * rank update accumulates the same terms into the same places in the same order.
     */
    @Test
    fun `splitting a panel into any group sizes computes the same thing`() {
        val rng = Random(20260920)
        val rows = 37
        val columns = 11
        val a = DoubleArray(rows * columns) { rng.nextDouble(-1.0, 1.0) }
        val x = DoubleArray(maxOf(rows, columns)) { rng.nextDouble(-1.0, 1.0) }
        val start = DoubleArray(maxOf(rows, columns)) { rng.nextDouble(-1.0, 1.0) }
        val kernels = koblas.panelKernels

        val whole = start.copyOf()
        kernels.multiDot(0.875, a, 0, rows, x, 0, 1, rows, columns, -0.25, whole, 0, 1)
        val wholeUpdate = start.copyOf()
        kernels.columnUpdate(0.875, a, 0, rows, x, 0, 1, rows, columns, wholeUpdate, 0, 1)

        for (group in intArrayOf(1, 2, 3, 5, 7, 16)) {
            val split = start.copyOf()
            forEachPanel(columns, group) { at, width ->
                kernels.multiDot(0.875, a, at * rows, rows, x, 0, 1, rows, width, -0.25, split, at, 1)
            }
            assertClose(whole, split, "multiDot split by $group")

            val splitUpdate = start.copyOf()
            forEachPanel(columns, group) { at, width ->
                kernels.columnUpdate(0.875, a, at * rows, rows, x, at, 1, rows, width, splitUpdate, 0, 1)
            }
            assertClose(wholeUpdate, splitUpdate, "columnUpdate split by $group")
        }
    }

    /** A non-positive recommendation would loop forever, so the shared traversal treats it as one column. */
    @Test
    fun `the shared traversal advances whatever grouping it is given`() {
        for (group in intArrayOf(-3, 0, 1, 4, 100)) {
            val widths = ArrayList<Int>()
            forEachPanel(7, group) { _, width -> widths.add(width) }
            assertEquals(7, widths.sum(), "group $group did not cover the panel")
            assertTrue(widths.all { it >= 1 }, "group $group produced an empty step")
        }
        val none = ArrayList<Int>()
        forEachPanel(0, 4) { _, width -> none.add(width) }
        assertTrue(none.isEmpty(), "an empty panel was visited")
    }

    private fun skipped() {
        println("SKIPPED: no Vector API panel candidate on this runtime; its bodies were not exercised")
    }
}
