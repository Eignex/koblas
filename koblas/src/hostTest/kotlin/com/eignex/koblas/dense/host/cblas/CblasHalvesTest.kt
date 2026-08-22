// The published-API detekt config for this custom source set wants KDoc everywhere and rejects backticked names.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.dense.host.cblas

import com.eignex.koblas.cblas.F64CblasLinearAlgebra
import com.eignex.koblas.cblas.F64CblasVectorKernels
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo
import com.eignex.koblas.registerBackend
import com.eignex.koblas.withCleanBackends
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** Checks that CBLAS and LAPACKE register independently of each other. */
class CblasHalvesTest {

    @Test
    fun `the level-1 kernels accept an empty run at the end of an array`() {
        val kernels = F64CblasVectorKernels()
        val v = doubleArrayOf(1.0, 2.0, 3.0)
        val end = v.size
        assertEquals(0.0, kernels.dot(v, end, v, end, 0), "dot")
        assertEquals(0.0, kernels.nrm2(v, end, 0), "nrm2")
        assertEquals(0.0, kernels.asum(v, end, 0), "asum")
        kernels.axpy(v, end, 2.0, v, end, 0)
        kernels.scale(v, end, 2.0, 0)
        assertEquals(listOf(1.0, 2.0, 3.0), v.toList(), "an empty run wrote something")
    }

    @Test
    fun `the BLAS half stands alone when LAPACKE is missing`() {
        val cblas = requireNotNull(OpenBlasLoader.cblas) { "host OpenBLAS expected in the test environment" }
        withCleanBackends {
            registerBackend(F64CblasBlas(cblas))
            assertEquals("cblas+reference", koblas.name, koblasInfo)

            val rng = kotlin.random.Random(20260804)
            val a = F64DenseMatrix.wrap(9, 9, DoubleArray(81) { rng.nextDouble(-1.0, 1.0) })
            val b = F64DenseMatrix.wrap(9, 9, DoubleArray(81) { rng.nextDouble(-1.0, 1.0) })
            val expected = F64ReferenceLinearAlgebra.gemm(a, b)
            val actual = koblas.gemm(a, b)
            for (i in expected.data.indices) {
                assertTrue(kotlin.math.abs(expected.data[i] - actual.data[i]) < 1e-12, "gemm entry $i")
            }

            val spd = F64DenseMatrix.diagonal(4, 2.0)
            assertEquals(4, koblas.factor(spd).n)
            assertEquals(2.0, koblas.cholesky(spd).l[0, 0] * koblas.cholesky(spd).l[0, 0], 1e-12)
        }
    }

    @Test
    fun `the LAPACK half stands alone too`() {
        val cblas = requireNotNull(OpenBlasLoader.cblas)
        val lapacke = requireNotNull(OpenBlasLoader.lapacke) { "host LAPACKE expected in the test environment" }
        withCleanBackends {
            registerBackend(F64CblasLapack(lapacke, cblas))
            assertEquals("reference+cblas", koblas.name, koblasInfo)
            assertEquals(4, koblas.factor(F64DenseMatrix.diagonal(4, 2.0)).n)
        }
    }

    @Test
    fun `both halves together land in both slots of the context`() {
        withCleanBackends {
            val whole: F64LinearAlgebra = F64CblasLinearAlgebra()
            registerBackend(whole)
            assertSame(whole, koblas.blas)
            assertSame(whole, koblas.lapack)
            assertEquals("cblas+reference", koblas.name)
        }
    }
}
