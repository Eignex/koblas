package com.eignex.koblas.sparse

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * The compiled-in sparse defaults, pinned so that collecting them cannot move a number a benchmark chose. A
 * run whose environment sets one of the keys will fail here, which is the intended reading.
 */
class SparseTuningTest {
    @Test
    fun `the sparse crossovers keep their measured values`() {
        assertEquals(16, SparseTuning.simdIndexedCrossover)
    }
}
