@file:Suppress("FunctionNaming")

package com.eignex.koblas

import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.NativeBlasPolicy
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Native crossover boundaries independent of an installed library. */
class NativeBlasPolicyTest {
    /** Whole and borrowed arrays share direct pinning economics. */
    @Test
    fun `native whole and borrowed calls use the same pinning crossover`() {
        for (case in listOf(
            NativeCase(DenseMatrixOperation.Gemm, 8, 8, 16),
            NativeCase(DenseMatrixOperation.Gemv, 32, 32, 1),
            NativeCase(DenseMatrixOperation.Syrk, 8, 8, 32),
            NativeCase(DenseMatrixOperation.Trsm, 16, 8, 16),
        )) {
            assertTrue(NativeBlasPolicy.eligible(case.operation, case.rows, case.columns, case.depth))
            assertTrue(NativeBlasPolicy.eligibleBorrowed(case.operation, case.rows, case.columns, case.depth))
            assertFalse(NativeBlasPolicy.eligible(case.operation, case.rows, case.columns - 1, case.depth))
        }
    }

    /** Empty work and unsupported windows never enter the native adapter. */
    @Test
    fun `empty work and unsupported windows stay owned`() {
        assertFalse(NativeBlasPolicy.eligible(DenseMatrixOperation.Gemm, 0, 128, 128))
        assertFalse(NativeBlasPolicy.eligible(DenseMatrixOperation.Gemm, 128, 128, 0))
        assertFalse(NativeBlasPolicy.eligible(DenseMatrixOperation.GerWindow, 128, 128, 1))
        assertTrue(NativeBlasPolicy.eligible(DenseMatrixOperation.Gemm, Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE))
    }

    private data class NativeCase(val operation: DenseMatrixOperation, val rows: Int, val columns: Int, val depth: Int)
}
