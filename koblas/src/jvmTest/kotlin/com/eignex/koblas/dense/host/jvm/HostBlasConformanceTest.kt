package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.dense.host.*
import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.cblas.OPENBLAS_SONAMES
import com.eignex.koblas.discoverBackends
import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.koblasInfo
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.test.*

@Category(HostLibraryTest::class)
class HostBlasConformanceTest {
    private fun requireCblas() {
        Assume.assumeTrue("host CBLAS is not installed", HostLibraries.cblas)
    }

    @Test
    fun `the host BLAS consumes strided views in place`() {
        requireCblas()
        assertStridedProductsAgreeWithReference(OpenBlas(HostBlasConfig()))
    }

    @Test
    fun `the host backend resolves when the machine has OpenBLAS`() {
        requireCblas()
        discoverBackends()
        assertTrue(koblasInfo.contains("openblas"), koblasInfo)
    }

    @Test
    fun `an LP64 host is not turned away by the width check`() {
        val library = FfmLibrary.open(OPENBLAS_SONAMES, "cblas_dgemm", "the host OpenBLAS")
        Assume.assumeTrue("host CBLAS is not installed", library.present)
        assertTrue(HostBlasCalls(HostBlasConfig()).available, "the host OpenBLAS was not recognized as LP64")
    }

    @Test
    fun `the gated level 2 and 3 routines match reference`() {
        requireCblas()
        assertTriangularAgreesWithReference(OpenBlas(), intArrayOf(1, 5, 12, 24))
        assertGerAgreesWithReference(OpenBlas())
        assertSyrAgreesWithReference(OpenBlas())
        assertSyr2kAgreesWithReference(OpenBlas(), intArrayOf(1, 5, 12, 24))
    }

    @Test
    fun `symv refuses a non-square matrix`() {
        requireCblas()
        assertSymvRefusesNonSquare(OpenBlas())
    }

    @Test
    fun `level 3 matches reference at blocked sizes`() {
        requireCblas()
        assertLevel3AgreesWithReference(OpenBlas(), intArrayOf(7, 64, 256))
    }

    @Test
    fun `the level 2 and 3 products match reference at moderate sizes`() {
        requireCblas()
        val host = OpenBlas()
        assertGemvAgreesWithReference(host, intArrayOf(18, 64))
        assertGemmAgreesWithReference(host, intArrayOf(18, 64))
        assertSyrkAgreesWithReference(host, intArrayOf(18, 64))
        assertSyrkTriangleModesLeaveTheOtherTriangle(host, intArrayOf(18, 64))
        assertSyr2kAgreesWithReference(host, intArrayOf(18, 64))
        assertSymmetricProductsAgreeWithReference(host, intArrayOf(18, 64))
    }

    @Test
    fun `the level 2 and 3 products match reference at small sizes`() {
        requireCblas()
        val host = OpenBlas()
        assertGemvAgreesWithReference(host, intArrayOf(7))
        assertGemmAgreesWithReference(host, intArrayOf(6))
        assertSyrkAgreesWithReference(host, intArrayOf(6))
        assertSyrkTriangleModesLeaveTheOtherTriangle(host, intArrayOf(6))
        assertSyr2kAgreesWithReference(host, intArrayOf(6))
        assertSymmetricProductsAgreeWithReference(host, intArrayOf(1, 6, 13))
    }

    @Test
    fun `degenerate shapes follow BLAS quick returns`() {
        requireCblas()
        assertDegenerateShapesFollowBlasQuickReturns(OpenBlas())
    }

    @Test
    fun `the binding constructs and reports unavailable without the library`() {
        val backend = OpenBlas(HostBlasConfig(libraryPath = MISSING_LIBRARY))
        assertFalse(backend.isAvailable, "a library that is not on this host cannot be available")
    }

    private companion object {
        const val MISSING_LIBRARY = "/nonexistent/libopenblas-koblas-test.so.0"
    }
}
