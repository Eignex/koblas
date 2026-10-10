package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PanelSchedulingTest {
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
}
