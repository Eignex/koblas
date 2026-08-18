package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DispatchThresholdsTest {

    @Test
    fun `the platform defaults are the ones this target was measured with`() {
        val defaults = platformDispatchThresholds
        // Scalar kernels never win level 1; SIMD kernels win level 2 at every size.
        if (mathBackend.startsWith("simd")) {
            assertTrue(defaults.level2 == Int.MAX_VALUE, "SIMD level 2 should stay portable, got ${defaults.level2}")
            assertTrue(defaults.level3 in 1..1024, "SIMD level 3 threshold looks wrong: ${defaults.level3}")
            assertTrue(defaults.lapack in 1..1024, "SIMD lapack threshold looks wrong: ${defaults.lapack}")
        } else {
            assertTrue(defaults.level3 == 0, "scalar kernels should dispatch level 3 from the start")
            assertTrue(defaults.level2 == 0, "scalar kernels should dispatch level 2 from the start")
        }
    }

    @Test
    fun `double precision owns the unqualified override keys and a later element type namespaces its own`() {
        assertEquals(
            "koblas.dispatch.level3",
            ConfigurationKeys.dispatchProperty(ElementType.F64, DispatchLevel.LEVEL3),
        )
        assertEquals(
            "KOBLAS_DISPATCH_LEVEL3",
            ConfigurationKeys.dispatchEnv(ElementType.F64, DispatchLevel.LEVEL3),
        )
        assertEquals(
            "koblas.dispatch.f32.level3",
            ConfigurationKeys.dispatchProperty(ElementType.F32, DispatchLevel.LEVEL3),
        )
        assertEquals(
            "KOBLAS_DISPATCH_F32_LEVEL3",
            ConfigurationKeys.dispatchEnv(ElementType.F32, DispatchLevel.LEVEL3),
        )
    }
}
