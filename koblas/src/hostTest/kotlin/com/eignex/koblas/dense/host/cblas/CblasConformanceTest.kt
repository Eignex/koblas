// The published-API detekt config for this custom source set wants KDoc everywhere and rejects backticked names.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.host.*
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import kotlin.test.*

/** Checks the CBLAS backend against the reference implementation. */
class CblasConformanceTest {
    private val cblas = F64CblasBackend()

    @Test
    fun `the native BLAS consumes strided views in place`() =
        assertStridedProductsAgreeWithReference(F64CblasBackend(HostBlasConfig()))

    @Test
    fun `discovery registers BLAS and install overrides it`() {
        assertTrue(F64CblasBackend.isAvailable(), "host OpenBLAS expected in the test environment")
        assertEquals("cblas", koblas.blas.name)
        assertEquals("reference", koblas.decompositions.name)
        try {
            installBackends(koblas.with(blas = F64ReferenceLinearAlgebra))
            assertEquals("reference", koblas.blas.name)
        } finally {
            installBackends(null)
        }
        assertEquals("cblas", koblas.blas.name)
        assertEquals("reference", koblas.decompositions.name)
    }

    @Test
    fun `gemv matches reference across transpose and alpha beta combos`() =
        assertGemvAgreesWithReference(cblas, intArrayOf(7))

    @Test
    fun `gemm matches reference across transpose and alpha beta combos`() =
        assertGemmAgreesWithReference(cblas, intArrayOf(6))

    @Test
    fun `syrk matches reference over the selected triangle`() = assertSyrkAgreesWithReference(cblas, intArrayOf(6))

    @Test
    fun `syrk triangle modes match reference and leave the other triangle untouched`() =
        assertSyrkTriangleModesLeaveTheOtherTriangle(cblas, intArrayOf(6))

    @Test
    fun `syr2k matches reference over both triangles`() = assertSyr2kAgreesWithReference(cblas, intArrayOf(6))

    @Test
    fun `symv and symm match reference with the unselected triangle poisoned`() =
        assertSymmetricProductsAgreeWithReference(cblas, intArrayOf(1, 6, 13))

    @Test
    fun `symv refuses a non-square matrix`() = assertSymvRefusesNonSquare(cblas)

    @Test
    fun `degenerate shapes follow BLAS quick returns`() = assertDegenerateShapesFollowBlasQuickReturns(cblas)

    @Test
    fun `level 3 agrees with the reference at blocked sizes`() =
        assertLevel3AgreesWithReference(cblas, intArrayOf(64, 256))

    @Test
    fun `triangular routines match reference across all flag combinations`() =
        assertTriangularAgreesWithReference(cblas, intArrayOf(1, 5, 12, 24))
}
