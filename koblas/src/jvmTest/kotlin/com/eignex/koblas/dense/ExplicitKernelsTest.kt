package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import kotlin.test.Test
import kotlin.test.assertEquals

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

    @Test
    fun `the bundled C symmetric product preserves finite cancellation`() {
        if (!JvmCKernelBindings.isAvailable) return
        assertSymvPreservesFiniteCancellation(CKernels)
    }

    @Test
    fun `the SIMD symmetric product preserves finite cancellation`() {
        if (!simdAvailable) return
        assertSymvPreservesFiniteCancellation(SimdKernels)
    }

    private fun assertSymvPreservesFiniteCancellation(kernels: Kernels) {
        val n = DenseTuning.symvFourColumnCrossover
        val column = n - 8
        val a = DenseMatrix(n, n)
        a[column + 1, column] = 1e308
        a[column + 2, column] = -1e308
        a[column + 4, column] = 1e308
        val x = DoubleArray(n)
        x[column + 1] = 1.0
        x[column + 2] = 1.0
        x[column + 4] = 1.0
        val expected = DoubleArray(n)
        val actual = DoubleArray(n)

        ReferenceBlas.symv(1.0, a, x, 0.0, expected, lower = true)
        BuiltinBlas(kernels).symv(1.0, a, x, 0.0, actual, lower = true)

        assertEquals(1e308, expected[column])
        assertEquals(expected[column], actual[column], kernels.name)
    }
}
