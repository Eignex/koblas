package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.assertCholeskyAgreesWithReference
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu
import com.eignex.koblas.sparse.sparseSymmetricConformanceSystem
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

@Category(HostLibraryTest::class)
class CholmodConformanceTest {

    // Gated at zero so these small systems reach the library instead of the portable factorization.
    private val umfpack = UmfpackSparseLu(UmfpackConfig(factorizeMin = 0))
    private val klu = KluSparseLu(KluConfig(factorizeMin = 0))

    private fun requireCholmod() {
        Assume.assumeTrue("CHOLMOD is not installed; conformance cannot run", CholmodCholesky().isAvailable)
    }

    private fun spd(n: Int, seed: Int) = sparseSymmetricConformanceSystem(n, Random(seed))

    @Test
    fun `the binding resolves on a machine with cholmod`() {
        requireCholmod()
        assertEquals(null, CholmodCholesky().unavailableReason)
    }

    /**
     * CHOLMOD has no LU and would take the seam from a backend that has one, so it fills the Cholesky of the
     * SuiteSparse backends rather than registering as one. Both of them have to answer with it.
     */
    @Test
    fun `both SuiteSparse backends answer the Cholesky with CHOLMOD`() {
        requireCholmod()
        Assume.assumeTrue("SuiteSparse is not installed", umfpack.isAvailable && klu.isAvailable)
        val a = spd(24, 20260912)

        assertIs<CholmodFactorization>(umfpack.cholesky(a), "UMFPACK should answer with CHOLMOD's Cholesky")
        assertIs<CholmodFactorization>(klu.cholesky(a), "KLU should answer with CHOLMOD's Cholesky")
    }

    @Test
    fun `solutions agree with the portable Cholesky`() {
        requireCholmod()
        Assume.assumeTrue("SuiteSparse is not installed", umfpack.isAvailable)
        assertCholeskyAgreesWithReference(umfpack)
    }

    @Test
    fun `an aliased destination still solves correctly`() {
        requireCholmod()
        Assume.assumeTrue("SuiteSparse is not installed", umfpack.isAvailable)
        val n = 18
        val a = spd(n, 20260913)
        val b = DoubleArray(n) { it * 0.25 - 1.0 }
        val f = umfpack.cholesky(a)
        val expected = f.solve(b)

        val aliased = b.copyOf()
        f.solveInto(aliased, aliased)

        assertClose(expected, aliased, "aliased destination", tolerance = 1e-12)
    }

    @Test
    fun `the reciprocal estimate is bounded and agrees in scale with the portable one`() {
        requireCholmod()
        Assume.assumeTrue("SuiteSparse is not installed", umfpack.isAvailable)
        val a = spd(20, 20260914)

        val host = umfpack.cholesky(a).rcond
        val portable = F64ReferenceSparseLinearAlgebra.cholesky(a).rcond

        assertTrue(host in 0.0..1.0, "estimate out of range: $host")
        // Different orderings give different factors, so this is about the quantity being the same one.
        assertTrue(host > portable / 100.0 && host < portable * 100.0, "host $host against portable $portable")
    }

    @Test
    fun `a matrix that is not positive definite is refused`() {
        requireCholmod()
        Assume.assumeTrue("SuiteSparse is not installed", umfpack.isAvailable)
        // Positive definite through column 1, then a diagonal too small to carry column 2.
        val indefinite = F64SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 4.0, 2 to 3.0), listOf(1 to 4.0), listOf(2 to 2.0)),
        )

        assertFailsWith<NotPositiveDefinite> { CholmodCholesky().factor(indefinite) }
    }

    @Test
    fun `the fill it reports counts the factor rather than the matrix`() {
        requireCholmod()
        Assume.assumeTrue("SuiteSparse is not installed", umfpack.isAvailable)
        val n = 24
        val a = spd(n, 20260915)

        val fill = umfpack.cholesky(a).nnz

        assertTrue(fill >= n, "a factor covers at least its own diagonal, got $fill")
    }
}
