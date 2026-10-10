package com.eignex.koblas.sparse

import kotlin.test.Test
import kotlin.test.assertEquals

/** Pins compiled-in sparse defaults. Environment overrides intentionally cause these checks to fail. */
class SparseTuningTest {
    @Test
    fun `the sparse crossovers keep their measured values`() {
        assertEquals(16, SparseTuning.simdIndexedCrossover)
    }
}
