package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import kotlin.test.Test

/**
 * [F64CKernels] directly, bypassing [F64PlatformKernels]'s automatic selection, so this backend is exercised
 * even on a JVM where SIMD shadows it for the platform-dispatched tests.
 */
class F64ExplicitKernelsTest {
    @Test
    fun `the bundled C kernels modified Givens agrees with the portable one`() {
        if (!JvmCKernelBindings.isAvailable) return
        assertModifiedGivensKernelsAgreeWithPortable(F64CKernels)
    }
}
