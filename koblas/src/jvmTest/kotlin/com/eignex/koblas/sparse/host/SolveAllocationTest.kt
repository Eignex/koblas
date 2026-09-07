package com.eignex.koblas.sparse.host

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.host.cholmod.CholmodCholesky
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.klu.KluFactorization
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackFactorization
import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.test.Test
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * What a native solve allocates, against the guarantee it declares.
 *
 * The conformance suites assert that a binding declares the size-independent guarantee and that a strict
 * solve is not rejected, which reads the declaration rather than the behaviour. These measure it. A wrapper
 * costs a fixed handful of bytes per call, so what has to hold is that the cost does not track the problem:
 * eight times the order is what makes that mean something, since a size-dependent cost would rise with it.
 */
@Category(HostLibraryTest::class)
class SolveAllocationTest {
    private val klu = KluSparseLu(KluConfig())
    private val umfpack = UmfpackSparseLu(UmfpackConfig())
    private val cholmod = CholmodCholesky()

    @Test
    fun `the KLU solve costs no more for a larger problem`() {
        Assume.assumeTrue("KLU is not installed", klu.isAvailable)

        assertSizeIndependent("KLU") { n ->
            assertIs<KluFactorization>(klu.factor(tridiagonal(n)), "measured a portable factorization")
        }
    }

    @Test
    fun `the UMFPACK solve costs no more for a larger problem`() {
        Assume.assumeTrue("SuiteSparse is not installed", umfpack.isAvailable)

        assertSizeIndependent("UMFPACK") { n ->
            assertIs<UmfpackFactorization>(umfpack.factor(tridiagonal(n)), "measured a portable factorization")
        }
    }

    @Test
    fun `the CHOLMOD solve costs no more for a larger problem`() {
        Assume.assumeTrue("CHOLMOD is not installed", cholmod.isAvailable)

        assertSizeIndependent("CHOLMOD") { n ->
            assertNotNull(cholmod.factor(tridiagonal(n, lowerOnly = true)), "CHOLMOD produced no factor")
        }
    }

    private fun assertSizeIndependent(label: String, factor: (Int) -> F64SparseFactorization) {
        val small = bytesPerSolve(SMALL, factor)
        val large = bytesPerSolve(LARGE, factor)

        assertTrue(
            large <= small * 2.0,
            "$label: $SMALL costs $small bytes a solve and $LARGE costs $large, which tracks the problem",
        )
        assertTrue(large < CEILING, "$label: $large bytes a solve at $LARGE is not a constant cost")
    }

    private fun bytesPerSolve(n: Int, factor: (Int) -> F64SparseFactorization): Double {
        val factorization = factor(n)
        val b = DoubleArray(n) { 1.0 + it }
        val out = DoubleArray(n)
        return bytesPerIteration(ITERATIONS, WARMUP, WINDOWS) { factorization.solveInto(b, out) }
    }

    /**
     * A tridiagonal diagonally dominant system, symmetric positive definite so one fixture serves the LU
     * bindings and the Cholesky one, as its lower triangle when [lowerOnly]. It factors in linear time at
     * any order, which the quarter-filled columns of the conformance systems would not at [LARGE].
     */
    private fun tridiagonal(n: Int, lowerOnly: Boolean = false): F64SparseMatrix = F64SparseMatrix.ofColumns(
        n,
        n,
        List(n) { j ->
            buildList {
                if (j > 0 && !lowerOnly) add(j - 1 to -1.0)
                add(j to 4.0)
                if (j < n - 1) add(j + 1 to -1.0)
            }
        },
    )

    private companion object {
        const val SMALL = 64
        const val LARGE = 512

        const val ITERATIONS = 2000
        const val WINDOWS = 3

        /**
         * Tiered compilation reaches the top tier somewhere around ten thousand invocations, and a figure
         * read below that is of code still on its way there rather than of the code a caller runs.
         */
        const val WARMUP = 20_000

        /** Comfortably above the few hundred bytes a wrapper costs, and far below anything tracking [LARGE]. */
        const val CEILING = 1024.0
    }
}
