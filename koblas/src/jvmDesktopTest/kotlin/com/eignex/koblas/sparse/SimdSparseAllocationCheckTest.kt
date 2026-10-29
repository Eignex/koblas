package com.eignex.koblas.sparse

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SimdSparseAllocationCheckTest {

    @Test
    fun `allocation workload crosses the default SIMD crossover`() {
        assertTrue(SimdSparseAllocationCheck.crossesSimdCrossover(16))
    }

    @Test
    fun `allocation workload rejects a crossover above its entry count`() {
        assertFalse(SimdSparseAllocationCheck.crossesSimdCrossover(SimdSparseAllocationCheck.ENTRY_COUNT + 1))
    }
}
