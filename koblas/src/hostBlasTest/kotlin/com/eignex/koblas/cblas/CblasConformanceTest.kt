// The published-API detekt config for this custom source set wants KDoc everywhere and rejects backticked names.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.cblas

import com.eignex.koblas.assertClose
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.assertAnEmptyFactorizationSolvesEmpty
import com.eignex.koblas.dense.assertDegenerateShapesHonorTheBetaConventions
import com.eignex.koblas.dense.assertDeterminantAgreesWithReference
import com.eignex.koblas.dense.assertFactorIntoUsesItsDestination
import com.eignex.koblas.dense.assertGemmAgreesWithReference
import com.eignex.koblas.dense.assertGemvAgreesWithReference
import com.eignex.koblas.dense.assertGerAgreesWithReference
import com.eignex.koblas.dense.assertLdlBlockSolveAgreesWithReference
import com.eignex.koblas.dense.assertLdlFactorsInterchange
import com.eignex.koblas.dense.assertLevel3AgreesWithReference
import com.eignex.koblas.dense.assertLuAgreesWithReference
import com.eignex.koblas.dense.assertLuFactorsInterchange
import com.eignex.koblas.dense.assertNonPositiveDefiniteFallsBack
import com.eignex.koblas.dense.assertQrFactorsInterchange
import com.eignex.koblas.dense.assertRcondAgreesWithReference
import com.eignex.koblas.dense.assertSingularLdlIsRefused
import com.eignex.koblas.dense.assertSingularLuIsFlagged
import com.eignex.koblas.dense.assertSpdSuiteAgreesWithReference
import com.eignex.koblas.dense.assertSymmetricProductsAgreeWithReference
import com.eignex.koblas.dense.assertSymvRefusesNonSquare
import com.eignex.koblas.dense.assertSyrkAgreesWithReference
import com.eignex.koblas.dense.assertSyrkTriangleModesLeaveTheOtherTriangle
import com.eignex.koblas.dense.assertTriangularAgreesWithReference
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Checks the CBLAS backend against the reference implementation. */
class CblasConformanceTest {

    private val cblas = CblasLinearAlgebra()

    @Test
    fun `discovery registers the backend and install overrides it`() {
        assertTrue(CblasLinearAlgebra.isAvailable(), "host OpenBLAS expected in the test environment")
        assertEquals("cblas", koblas.blas.name)
        try {
            installBackends(koblas.with(blas = ReferenceLinearAlgebra, lapack = ReferenceLinearAlgebra))
            assertEquals("reference", koblas.blas.name)
            assertEquals("reference", koblas.lapack.name)
        } finally {
            installBackends(null) // restores automatic selection
        }
        assertEquals("cblas", koblas.blas.name)
    }

    @Test
    fun `gemv matches reference across transpose and alpha beta combos`() =
        assertGemvAgreesWithReference(cblas, intArrayOf(7))

    @Test
    fun `gemm matches reference across transpose and alpha beta combos`() =
        assertGemmAgreesWithReference(cblas, intArrayOf(6))

    @Test
    fun `syrk matches reference and is exactly symmetric`() = assertSyrkAgreesWithReference(cblas, intArrayOf(6))

    @Test
    fun `syrk triangle modes match reference and leave the other triangle untouched`() =
        assertSyrkTriangleModesLeaveTheOtherTriangle(cblas, intArrayOf(6))

    @Test
    fun `symv and symm match reference with the unselected triangle poisoned`() =
        assertSymmetricProductsAgreeWithReference(cblas, intArrayOf(1, 6, 13))

    @Test
    fun `symv refuses a non-square matrix`() = assertSymvRefusesNonSquare(cblas)

    @Test
    fun `the LU family matches reference in both directions and for a block`() =
        assertLuAgreesWithReference(cblas, intArrayOf(1, 3, 8, 33))

    @Test
    fun `factorInto refactorizes into the destination it was given`() =
        assertFactorIntoUsesItsDestination(cblas, n = 24)

    @Test
    fun `the determinant matches reference`() = assertDeterminantAgreesWithReference(cblas, intArrayOf(1, 3, 8, 33))

    @Test
    fun `ldl block solves match reference`() = assertLdlBlockSolveAgreesWithReference(cblas, n = 9, nrhs = 4)

    @Test
    fun `a singular LDL is refused at every width`() = assertSingularLdlIsRefused(cblas)

    @Test
    fun `factorizations interchange between backends`() = assertLuFactorsInterchange(cblas, n = 12)

    @Test
    fun `rcond agrees with the reference estimator in magnitude`() =
        assertRcondAgreesWithReference(cblas, intArrayOf(1, 6, 24))

    @Test
    fun `qr factorizations interchange between backends`() = assertQrFactorsInterchange(cblas, listOf(6 to 6, 10 to 4))

    @Test
    fun `ldl factorizations match and interchange between backends`() =
        assertLdlFactorsInterchange(cblas, intArrayOf(1, 2, 5, 14, 33))

    @Test
    fun `singular matrix sets the flag and zero determinant`() = assertSingularLuIsFlagged(cblas)

    @Test
    fun `degenerate shapes honor the beta conventions`() = assertDegenerateShapesHonorTheBetaConventions(cblas)

    @Test
    fun `an empty factorization solves an empty right-hand side`() = assertAnEmptyFactorizationSolvesEmpty(cblas)

    /** The rest of the suite stays below 50, where OpenBLAS is serial and unblocked. 64 and 256 cross into both. */
    @Test
    fun `level 3 agrees with the reference at blocked sizes`() =
        assertLevel3AgreesWithReference(cblas, intArrayOf(64, 256))

    @Test
    fun `triangular routines match reference across all flag combinations`() =
        assertTriangularAgreesWithReference(cblas, intArrayOf(1, 5, 12, 24))

    @Test
    fun `the SPD suite matches reference`() = assertSpdSuiteAgreesWithReference(cblas, intArrayOf(1, 4, 9, 33))

    @Test
    fun `a non positive definite input falls back to the portable path`() =
        assertNonPositiveDefiniteFallsBack(cblas, n = 2)

    @Test
    fun `ger matches reference`() = assertGerAgreesWithReference(cblas)

    /** Lengths 63, 64 and 65 straddle the routing threshold, so both the host and the scalar path are covered. */
    @Test
    fun `installed level-1 kernels agree with the scalar ones`() {
        val kernels = CblasVectorKernels()
        val rng = Random(20260731)
        for (len in intArrayOf(1, 7, 63, 64, 65, 200)) {
            val pad = 3 // a non-zero offset, so an implementation that ignores it fails
            val a = DoubleArray(len + pad) { rng.nextDouble(-1.0, 1.0) }
            val b = DoubleArray(len + pad) { rng.nextDouble(-1.0, 1.0) }
            var expectedDot = 0.0
            for (i in 0 until len) expectedDot += a[pad + i] * b[i]
            assertClose(
                doubleArrayOf(expectedDot),
                doubleArrayOf(kernels.dot(a, pad, b, 0, len)),
                context = "dot len=$len",
            )
            val expectedAxpy = b.copyOf()
            for (i in 0 until len) expectedAxpy[i] += 0.75 * a[pad + i]
            val actualAxpy = b.copyOf()
            kernels.axpy(actualAxpy, 0, 0.75, a, pad, len)
            assertClose(expectedAxpy, actualAxpy, context = "axpy len=$len")
            val expectedScale = a.copyOf()
            for (i in 0 until len) expectedScale[pad + i] *= -0.5
            val actualScale = a.copyOf()
            kernels.scale(actualScale, pad, -0.5, len)
            assertClose(expectedScale, actualScale, context = "scale len=$len")
        }
    }

    @Test
    fun `the routed reductions agree with the built-in ones`() {
        val kernels = CblasVectorKernels()
        val rng = Random(20260951)
        for (scale in doubleArrayOf(1.0, 1e200, 1e-200)) {
            for (len in intArrayOf(1, 63, 64, 200)) {
                val pad = 3 // a non-zero offset, so an implementation that ignores it fails
                val v = DoubleArray(len + pad) { rng.nextDouble(-1.0, 1.0) * scale }
                val ctx = "len=$len scale=$scale"
                assertClose(
                    doubleArrayOf(referenceNrm2(v, pad, len)),
                    doubleArrayOf(kernels.nrm2(v, pad, len)),
                    context = "nrm2 $ctx",
                )
                var expectedAsum = 0.0
                for (i in 0 until len) expectedAsum += abs(v[pad + i])
                assertClose(
                    doubleArrayOf(expectedAsum),
                    doubleArrayOf(kernels.asum(v, pad, len)),
                    context = "asum $ctx",
                )
            }
        }
        val zeros = DoubleArray(80)
        assertEquals(0.0, kernels.nrm2(zeros, 0, 80), "nrm2 of zeros")
        assertEquals(0.0, kernels.asum(zeros, 0, 80), "asum of zeros")
    }

    /** The rescaled two-pass norm, written out here so the reference does not use the implementation under test. */
    private fun referenceNrm2(v: DoubleArray, off: Int, len: Int): Double {
        var amax = 0.0
        for (i in 0 until len) {
            val a = abs(v[off + i])
            if (a > amax) amax = a
        }
        if (amax == 0.0) return 0.0
        var t = 0.0
        for (i in 0 until len) {
            val r = v[off + i] / amax
            t += r * r
        }
        return amax * sqrt(t)
    }
}
