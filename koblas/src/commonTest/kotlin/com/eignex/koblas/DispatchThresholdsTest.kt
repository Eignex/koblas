package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertTrue

class DispatchThresholdsTest {

    @Test
    fun `thresholds are non-negative and ordered by how much work amortizes a call`() {
        val t = dispatchThresholds
        assertTrue(t.level2 >= 0 && t.level3 >= 0 && t.lapack >= 0, "negative threshold: $t")
        assertTrue(t.level2 >= t.level3, "level 2 (${t.level2}) should not dispatch before level 3 (${t.level3})")
    }

    @Test
    fun `the platform defaults are the ones this target was measured with`() {
        val defaults = platformDispatchThresholds
        // Scalar kernels lose from the smallest size measured, SIMD kernels win level 2 at every size.
        if (mathBackend.startsWith("simd")) {
            assertTrue(defaults.level2 == Int.MAX_VALUE, "SIMD level 2 should stay portable, got ${defaults.level2}")
            assertTrue(defaults.level3 in 1..1024, "SIMD level 3 threshold looks wrong: ${defaults.level3}")
            assertTrue(defaults.lapack in 1..1024, "SIMD lapack threshold looks wrong: ${defaults.lapack}")
        } else {
            assertTrue(defaults.level3 == 0, "scalar kernels should dispatch level 3 from the start")
            assertTrue(defaults.level2 == 0, "scalar kernels should dispatch level 2 from the start")
        }
    }
}
