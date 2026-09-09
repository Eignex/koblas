package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import kotlin.test.Test

/**
 * [CKernels] and [SimdKernels] directly, bypassing [PlatformKernels]'s automatic selection, so both
 * backends are exercised even on a JVM where one shadows the other for the platform-dispatched tests.
 */
class ExplicitKernelsTest {
    @Test
    fun `the bundled C kernels modified Givens agrees with the portable one`() {
        if (!JvmCKernelBindings.isAvailable) return
        assertModifiedGivensKernelsAgreeWithPortable(CKernels)
        assertRotKernelAgreesWithPortable(CKernels)
    }

    @Test
    fun `the SIMD kernels modified Givens agrees with the portable one`() {
        if (!simdAvailable) return
        assertModifiedGivensKernelsAgreeWithPortable(SimdKernels)
        assertRotKernelAgreesWithPortable(SimdKernels)
    }
}
