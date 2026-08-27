package com.eignex.koblas.openblas

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.host.cblas.OpenBlasOptions
import kotlin.test.*

class BundledOpenBlasTest {
    @Test
    fun `registers through the backend service`() {
        discoverBackends()
        assertEquals("openblas-bundled", koblas.blas.name)
        assertEquals("openblas-bundled", koblas.decompositions.name)
    }

    @Test
    fun `loads the Maven OpenBLAS binary`() {
        val backend = BundledOpenBlas()

        assertTrue(backend.isAvailable)
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 4.0)))
        val out = F64DenseMatrix.zero(2)
        backend.gemm(1.0, a, false, a, false, 0.0, out)

        assertEquals(listOf(7.0, 10.0, 15.0, 22.0), out.data.toList())
    }

    @Test
    fun `shared options control bundled routing and diagnostics`() {
        val backend = BundledOpenBlas(
            OpenBlasOptions(threadCount = 1, level3Min = 64, factorizeMin = 48),
        )

        val route = assertNotNull(backend.route(F64RouteQuery.DenseGemm(m = 8, n = 8, k = 8)))

        assertEquals(BackendExecution.PORTABLE, route.execution)
        assertEquals("64", backend.backendMetadata.options["level3Min"])
        assertEquals("48", backend.backendMetadata.options["factorizeMin"])
        assertEquals("1 threads", backend.backendMetadata.threading)
    }
}
