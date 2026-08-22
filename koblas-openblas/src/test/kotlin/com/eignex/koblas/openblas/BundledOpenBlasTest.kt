package com.eignex.koblas.openblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BundledOpenBlasTest {
    @Test
    fun `registers through the backend service`() {
        assertEquals("openblas-bundled", koblas.blas.name)
        assertEquals("openblas-bundled", koblas.lapack.name)
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
}
