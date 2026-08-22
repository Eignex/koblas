// The published-API detekt config for this custom source set wants KDoc everywhere and rejects backticked names.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.cblas.F64CblasLinearAlgebra
import com.eignex.koblas.cblas.F64CblasVectorKernels
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.assertLevel1KernelsAgreeWithScalar
import com.eignex.koblas.dense.assertReductionsAgreeWithScalar
import com.eignex.koblas.dense.host.*
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Checks the CBLAS backend against the reference implementation. */
class CblasConformanceTest {

    private val cblas = F64CblasLinearAlgebra()

    @Test
    fun `discovery registers the backend and install overrides it`() {
        assertTrue(F64CblasLinearAlgebra.isAvailable(), "host OpenBLAS expected in the test environment")
        assertEquals("cblas", koblas.blas.name)
        try {
            installBackends(koblas.with(blas = F64ReferenceLinearAlgebra, lapack = F64ReferenceLinearAlgebra))
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

    /** Native binds dgeqp3 optionally, so this is a real comparison only where the host provides it. */
    @Test
    fun `pivoted QR matches reference in rank and reconstruction`() =
        assertPivotedQrAgreesWithReference(cblas, m = 40, cols = 24, ranks = intArrayOf(24, 12))

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

    @Test
    fun `installed level-1 kernels agree with the scalar ones`() =
        assertLevel1KernelsAgreeWithScalar(F64CblasVectorKernels())

    @Test
    fun `the routed reductions agree with the built-in ones`() = assertReductionsAgreeWithScalar(
        F64CblasVectorKernels(),
    )
}
