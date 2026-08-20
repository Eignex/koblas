package com.eignex.koblas.dense

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.f64DispatchThresholds
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Discovery accepts a candidate on the strength of one probe call, so the probe has to reach the native path
 * the candidate would use in anger rather than the fallback it keeps for small problems.
 */
class DiscoveryProbeTest {

    /** A backend shaped like koblas's own adapters: portable below its gate, native at or above it. */
    private class GatedBackend(private val nativeWorks: Boolean) : F64Blas by F64ReferenceLinearAlgebra {
        override val name: String get() = "gated"

        @Suppress("LongParameterList") // the BLAS dgemm signature
        override fun gemm(
            alpha: Double,
            a: F64DenseMatrix,
            transposeA: Boolean,
            b: F64DenseMatrix,
            transposeB: Boolean,
            beta: Double,
            c: F64DenseMatrix,
        ) {
            if (a.rows >= f64DispatchThresholds.level3 && !nativeWorks) {
                throw UnsatisfiedLinkError("libgated.so: cannot open shared object file")
            }
            F64ReferenceLinearAlgebra.gemm(alpha, a, transposeA, b, transposeB, beta, c)
        }
    }

    @Test
    fun `a candidate whose native path is missing is rejected`() {
        assertFalse(probe(GatedBackend(nativeWorks = false)))
    }

    @Test
    fun `a candidate whose native path works is accepted`() {
        assertTrue(probe(GatedBackend(nativeWorks = true)))
    }

    @Test
    fun `the reference backend passes its own probe`() {
        assertTrue(probe(F64ReferenceBackend()))
    }
}
