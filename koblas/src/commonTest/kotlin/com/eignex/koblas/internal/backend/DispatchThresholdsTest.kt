package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import kotlin.test.*

class DispatchThresholdsTest {

    /** Kernels whose name is nothing like the compiled-in ones, so a threshold that reads it would move. */
    private class ForeignKernels : F64Kernels {
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
        installBackends(koblas.with(kernels = ForeignKernels()))
        try {
            val during = platformDispatchThresholds
            assertEquals(before.level1, during.level1, "level1 followed the installed backend")
            assertEquals(before.level2, during.level2, "level2 followed the installed backend")
            assertEquals(before.level3, during.level3, "level3 followed the installed backend")
            assertEquals(before.factorize, during.factorize, "factorize followed the installed backend")
        } finally {
            installBackends(null)
        }
        assertEquals(before.factorize, platformDispatchThresholds.factorize, "factorize did not come back")
    }

    /**
     * The gate is measured as the dimension at which a cube crosses over, and the work at that crossover is
     * its cube. A shape that reaches the same work has paid for the crossing however narrow one of its sides
     * is, which the narrowest dimension alone cannot say.
     */
    @Test
    fun `a level-3 gate follows the work rather than the narrowest dimension`() {
        val gate = 64

        assertTrue(dispatchesLevel3(gate, gate, gate, gate), "the cube it was measured as")
        assertFalse(dispatchesLevel3(gate, gate - 1, gate, gate), "just under the crossover")
        assertTrue(dispatchesLevel3(gate, 4096, 2, 4096), "a large product of a two-column matrix")
        assertFalse(dispatchesLevel3(gate, 4096, 2, 8), "a small one of the same shape")
    }

    @Test
    fun `a level-3 gate is never less permissive than the dimension it is measured as`() {
        for (gate in intArrayOf(0, 1, 16, 64, 1024)) {
            for (m in intArrayOf(1, 2, 63, 64, 1024)) {
                for (n in intArrayOf(1, 2, 63, 64, 1024)) {
                    for (k in intArrayOf(1, 2, 63, 64, 1024)) {
                        if (minOf(m, n, k) < gate) continue
                        assertTrue(dispatchesLevel3(gate, m, n, k), "gate=$gate shape=${m}x${n}x$k")
                    }
                }
            }
        }
    }

    @Test
    fun `a level-3 family pinned portable stays portable at any size`() {
        assertFalse(dispatchesLevel3(Int.MAX_VALUE, 1, 1, 1))
        assertFalse(dispatchesLevel3(Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE))
    }

    @Test
    fun `the platform defaults are the ones this target was measured with`() {
        val defaults = platformDispatchThresholds
        // Scalar kernels never win level 1; SIMD kernels win level 2 at every size.
        if (mathBackend.startsWith("simd")) {
            assertEquals(
                Int.MAX_VALUE,
                defaults.level2,
                "SIMD level 2 should stay portable, got ${defaults.level2}",
            )
            assertTrue(defaults.level3 in 1..1024, "SIMD level 3 threshold looks wrong: ${defaults.level3}")
            assertTrue(
                defaults.factorize in 1..1024,
                "SIMD factorize threshold looks wrong: ${defaults.factorize}",
            )
        } else {
            assertEquals(
                0,
                defaults.level3,
                "scalar kernels should dispatch level 3 from the start",
            )
            assertEquals(
                0,
                defaults.level2,
                "scalar kernels should dispatch level 2 from the start",
            )
        }
    }
}
