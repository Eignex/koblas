package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.assertCholeskyFactorReproduces
import com.eignex.koblas.sparse.assertLdlFactorsReproduce
import com.eignex.koblas.sparse.sparseSymmetricConformanceSystem
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull

@Category(HostLibraryTest::class)
class CholmodFactorsTest {
    private val cholmod = CholmodCholesky()

    private fun requireCholmod() {
        Assume.assumeTrue("CHOLMOD is not installed", cholmod.isAvailable)
    }

    @Test
    fun `the Cholesky factor reproduces the permuted matrix`() {
        requireCholmod()
        val rng = Random(20260827)
        for (n in intArrayOf(4, 20, 80)) {
            val a = sparseSymmetricConformanceSystem(n, rng, density = 0.2)

            assertNotNull(cholmod.factor(a)).use { factorization ->
                assertCholeskyFactorReproduces(a, factorization, "n=$n")
            }
        }
    }

    @Test
    fun `the LDL factors reproduce the permuted matrix`() {
        requireCholmod()
        val rng = Random(20260901)
        for (n in intArrayOf(4, 20, 80)) {
            val a = sparseSymmetricConformanceSystem(n, rng, density = 0.2)

            assertNotNull(cholmod.factorQuasiDefiniteLdl(a)).use { factorization ->
                assertLdlFactorsReproduce(a, factorization, "n=$n")
            }
        }
    }

    @Test
    fun `a singular LDL factorization has no factors to give`() {
        requireCholmod()
        val a = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), emptyList()))
        val factorization = assertNotNull(cholmod.factorQuasiDefiniteLdl(a))

        assertFailsWith<com.eignex.koblas.SingularMatrix> { factorization.l }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { factorization.d }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { factorization.order }
    }
}
