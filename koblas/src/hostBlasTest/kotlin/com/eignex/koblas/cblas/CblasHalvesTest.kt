// hostBlasTest is a custom source set, which the shared convention plugin analyses with the config for
// published API rather than for tests. The suite is documented at class level like the rest.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.cblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo
import com.eignex.koblas.registerBackend
import com.eignex.koblas.withCleanBackends
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * The two halves of the host backend, installed separately.
 *
 * Debian and Ubuntu ship LAPACKE as a package that libopenblas0 does not pull in, so CBLAS present with
 * LAPACKE absent is a common configuration rather than an edge case. It cannot be produced on a host that
 * has both, so what these tests pin is the mechanism it relies on: that the BLAS half works standalone,
 * that koblas composes it with the portable factorizations, and that level-3 work still reaches the host
 * library in that state. Without this, a regression to all-or-nothing would go unnoticed until someone
 * on such a host wondered why their matrices were slow.
 */
class CblasHalvesTest {

    @Test
    fun `the BLAS half stands alone when LAPACKE is missing`() {
        val cblas = requireNotNull(OpenBlasLoader.cblas) { "host OpenBLAS expected in the test environment" }
        withCleanBackends {
            registerBackend(CblasBlas(cblas))
            assertEquals("cblas+reference", koblas.name, koblasInfo)

            // Level 3 reaches the host library.
            val rng = kotlin.random.Random(20260804)
            val a = DenseMatrix.wrap(9, 9, DoubleArray(81) { rng.nextDouble(-1.0, 1.0) })
            val b = DenseMatrix.wrap(9, 9, DoubleArray(81) { rng.nextDouble(-1.0, 1.0) })
            val expected = ReferenceLinearAlgebra.gemm(a, b)
            val actual = koblas.gemm(a, b)
            for (i in expected.data.indices) {
                assertTrue(kotlin.math.abs(expected.data[i] - actual.data[i]) < 1e-12, "gemm entry $i")
            }

            // And the factorizations answer from the portable side rather than failing.
            val spd = DenseMatrix.diagonal(4, 2.0)
            assertEquals(4, koblas.factor(spd).n)
            assertEquals(2.0, koblas.cholesky(spd)[0, 0] * koblas.cholesky(spd)[0, 0], 1e-12)
        }
    }

    @Test
    fun `the LAPACK half stands alone too`() {
        val cblas = requireNotNull(OpenBlasLoader.cblas)
        val lapacke = requireNotNull(OpenBlasLoader.lapacke) { "host LAPACKE expected in the test environment" }
        withCleanBackends {
            registerBackend(CblasLapack(lapacke, cblas))
            assertEquals("reference+cblas", koblas.name, koblasInfo)
            assertEquals(4, koblas.factor(DenseMatrix.diagonal(4, 2.0)).n)
        }
    }

    @Test
    fun `both halves together land in both slots of the context`() {
        withCleanBackends {
            val whole: LinearAlgebra = CblasLinearAlgebra()
            registerBackend(whole)
            // One object, offered as every half it implements: both dense slots point at it, and the sparse
            // ones stay on the reference, which the name reports.
            assertSame(whole, koblas.blas)
            assertSame(whole, koblas.lapack)
            assertEquals("cblas+reference", koblas.name)
        }
    }
}
