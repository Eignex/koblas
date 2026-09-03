package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import kotlin.test.Test

/**
 * [F64CKernels] and [F64SimdKernels] directly, bypassing [F64PlatformKernels]'s automatic selection, so both
 * backends are exercised even on a JVM where one shadows the other for the platform-dispatched tests.
 */
class F64ExplicitKernelsTest {
    @Test
    fun `the bundled C kernels modified Givens agrees with the portable one`() {
        if (!JvmCKernelBindings.isAvailable) return
        assertModifiedGivensKernelsAgreeWithPortable(F64CKernels)
        assertRotKernelAgreesWithPortable(F64CKernels)
    }

    @Test
    fun `the SIMD kernels modified Givens agrees with the portable one`() {
        if (!simdAvailable) return
        assertModifiedGivensKernelsAgreeWithPortable(F64SimdKernels)
        assertRotKernelAgreesWithPortable(F64SimdKernels)
    }
}
