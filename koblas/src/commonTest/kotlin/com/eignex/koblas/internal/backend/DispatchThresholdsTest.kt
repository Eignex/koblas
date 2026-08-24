package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.dense.F64VectorKernels
import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DispatchThresholdsTest {

    /** Kernels whose name is nothing like the compiled-in ones, so a threshold that reads it would move. */
    private class ForeignKernels : F64VectorKernels {
        override val name: String get() = "foreign"
        override val priority: Int get() = 100
        override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
            var s = 0.0
            for (i in 0 until len) s += a[aOff + i] * b[bOff + i]
            return s
        }
        override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
            for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
        }
        override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
            for (i in 0 until len) v[vOff + i] *= alpha
        }
        override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = euclideanNorm(v, vOff, len)
        override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)
    }

    @Test
    fun `the platform defaults describe the target rather than the installed backend`() {
        // f64DispatchThresholds memoizes these, so anything they read must hold for the whole process.
        // Reading an installed backend instead would freeze whatever happened to be active at the first
        // read and leave every later dispatch deciding against the wrong table.
        val before = platformDispatchThresholds
        installBackends(koblas.with(vectorKernels = ForeignKernels()))
        try {
            val during = platformDispatchThresholds
            assertEquals(before.level1, during.level1, "level1 followed the installed backend")
            assertEquals(before.level2, during.level2, "level2 followed the installed backend")
            assertEquals(before.level3, during.level3, "level3 followed the installed backend")
            assertEquals(before.factorize, during.factorize, "factorize followed the installed backend")
            assertEquals(before.factorizeRhs, during.factorizeRhs, "factorizeRhs followed the installed backend")
        } finally {
            installBackends(null)
        }
        assertEquals(before.factorize, platformDispatchThresholds.factorize, "factorize did not come back")
    }

    @Test
    fun `the platform defaults are the ones this target was measured with`() {
        val defaults = platformDispatchThresholds
        // Scalar kernels never win level 1; SIMD kernels win level 2 at every size.
        if (mathBackend.startsWith("simd")) {
            assertTrue(defaults.level2 == Int.MAX_VALUE, "SIMD level 2 should stay portable, got ${defaults.level2}")
            assertTrue(defaults.level3 in 1..1024, "SIMD level 3 threshold looks wrong: ${defaults.level3}")
            assertTrue(
                defaults.factorize in 1..1024,
                "SIMD factorize threshold looks wrong: ${defaults.factorize}",
            )
        } else {
            assertTrue(defaults.level3 == 0, "scalar kernels should dispatch level 3 from the start")
            assertTrue(defaults.level2 == 0, "scalar kernels should dispatch level 2 from the start")
        }
    }
}
