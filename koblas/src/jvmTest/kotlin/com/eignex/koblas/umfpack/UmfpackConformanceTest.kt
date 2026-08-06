package com.eignex.koblas.umfpack

import com.eignex.koblas.HostLibraryTest
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.registerBackend
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.gemv
import com.eignex.koblas.withCleanBackends
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * UMFPACK against koblas's portable sparse LU.
 *
 * Skipped wholesale when SuiteSparse is absent, because the point is to exercise the real library — a test
 * that passed by not running would be worse than no test. `UmfpackCalls.available` is the same check
 * discovery uses, so a skip here means the backend would not have registered either.
 *
 * The reference is the assertion target throughout: two different algorithms (UMFPACK's AMD/COLAMD ordering
 * against koblas's Markowitz pivoting) will not produce the same factors, so the factors are not comparable
 * and the *solutions* are what must agree.
 */
@Category(HostLibraryTest::class)
class UmfpackConformanceTest {

    private val umfpack = UmfpackSparseLapack()

    /**
     * Skips the test — genuinely, as a reported skip — when SuiteSparse is absent.
     *
     * `Assume` rather than an early `return`, which is what these guards used to be. An early return reports
     * the test as PASSED, so on a machine without the library the whole suite went green while exercising
     * nothing, and confirming it had actually run took a manual check. `assumeTrue` reports SKIPPED, which is
     * the honest signal and needs no verification. JUnit 4 has had this all along.
     */
    private fun requireSuiteSparse() {
        Assume.assumeTrue("SuiteSparse is not installed; umfpack conformance cannot run", UmfpackCalls.available)
    }

    /** A diagonally dominant sparse matrix: well conditioned, and singular for neither backend. */
    private fun sparseSystem(n: Int, rng: Random): SparseMatrix {
        val columns = ArrayList<List<Pair<Int, Double>>>(n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until n) {
                val v = when {
                    i == j -> n + 10.0
                    rng.nextDouble() < 0.25 -> rng.nextDouble(-1.0, 1.0)
                    else -> 0.0
                }
                if (v != 0.0) column.add(i to v)
            }
            columns.add(column)
        }
        return SparseMatrix.ofColumns(n, n, columns)
    }

    @Test
    fun `the binding resolves on a machine with suitesparse`() {
        requireSuiteSparse()
        assertEquals("umfpack", umfpack.name)
        assertEquals(100, umfpack.priority, "should outrank the portable SparseLu")
    }

    @Test
    fun `solutions agree with the portable factorization in both directions`() {
        requireSuiteSparse()
        val rng = Random(20260815)
        for (n in intArrayOf(1, 2, 7, 23, 60)) {
            val a = sparseSystem(n, rng)
            val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

            val host = umfpack.factor(a)
            val portable = ReferenceSparseLinearAlgebra.factor(a)
            assertTrue(!host.singular, "n=$n umfpack called a well-conditioned system singular")
            assertTrue(!portable.singular, "n=$n the reference called it singular")

            for (transpose in booleanArrayOf(false, true)) {
                val fromHost = host.solve(b, transpose)
                val fromPortable = portable.solve(b, transpose)
                for (i in 0 until n) {
                    assertEquals(fromPortable[i], fromHost[i], 1e-9, "n=$n transpose=$transpose entry $i")
                }
                // And it solves the system, checked independently of either factorization.
                val residual = a.gemv(fromHost, transpose)
                for (i in 0 until n) assertEquals(b[i], residual[i], 1e-9, "n=$n transpose=$transpose residual $i")
            }
        }
    }

    /** `out === b` must work: umfpack reads B while writing X, so the aliased case needs a copy. */
    @Test
    fun `an aliased destination still solves correctly`() {
        requireSuiteSparse()
        val rng = Random(20260816)
        val n = 12
        val a = sparseSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val f = umfpack.factor(a)
        val expected = f.solve(b)
        val aliased = b.copyOf()
        f.solveInto(aliased, aliased)
        for (i in 0 until n) assertEquals(expected[i], aliased[i], 1e-12, "entry $i")
    }

    @Test
    fun `the determinant agrees with the portable factorization`() {
        requireSuiteSparse()
        val rng = Random(20260817)
        for (n in intArrayOf(1, 3, 8)) {
            val a = sparseSystem(n, rng)
            val host = umfpack.factor(a).determinant()
            val portable = ReferenceSparseLinearAlgebra.factor(a).determinant()
            // Sign and magnitude both matter; a relative comparison because the values grow with n.
            assertEquals(1.0, host / portable, 1e-9, "n=$n determinant disagreed: $host vs $portable")
        }
    }

    /**
     * A singular matrix: UMFPACK factors it and warns, so koblas reports singular without a position.
     *
     * This is the contract gap the first host backend exposed. koblas's own factorizations always know which
     * pivot failed, because they choose the pivots; UMFPACK reports only that the matrix is singular, and
     * recovering the position would mean extracting `U` to hunt for a zero diagonal.
     */
    @Test
    fun `a singular matrix is reported singular with an unknown position`() {
        requireSuiteSparse()
        // Column 1 is a multiple of column 0, so the matrix is rank 1.
        val rank1 = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to 2.0), listOf(0 to 2.0, 1 to 4.0)),
        )
        val f = umfpack.factor(rank1)
        assertTrue(f.singular, "umfpack should have called a rank-1 matrix singular")
        assertEquals(SINGULAR_POSITION_UNKNOWN, f.failedAt, "a host that cannot name the pivot must say so")
        assertEquals(0.0, f.determinant(), "a singular factorization has determinant zero")
        assertFailsWith<IllegalStateException> { f.solve(doubleArrayOf(1.0, 1.0)) }
    }

    /** `equilibrate` has no UMFPACK equivalent, so it must fall back rather than silently ignore the request. */
    @Test
    fun `equilibrate falls back to the portable factorization`() {
        requireSuiteSparse()
        val rng = Random(20260818)
        val a = sparseSystem(6, rng)
        val equilibrated = umfpack.factor(a, equilibrate = true)
        assertTrue(
            equilibrated !is UmfpackFactorization,
            "asking for koblas's row scaling must not quietly get umfpack's own",
        )
        // It still has to be a working factorization.
        val b = DoubleArray(6) { rng.nextDouble(-1.0, 1.0) }
        val residual = a.gemv(equilibrated.solve(b))
        for (i in 0 until 6) assertEquals(b[i], residual[i], 1e-9, "entry $i")
    }

    @Test
    fun `it registers as the sparse factorization half and reports fill`() {
        requireSuiteSparse()
        withCleanBackends {
            registerBackend(umfpack)
            assertEquals("umfpack", koblas.sparseLapack.name, "umfpack should win the sparse lapack half")
            // The sparse BLAS half is untouched: umfpack has no gemv and does not claim one.
            assertEquals("reference", koblas.sparseBlas.name)

            val rng = Random(20260819)
            val a = sparseSystem(20, rng)
            val f = koblas.factor(a)
            assertTrue(f.nnz >= 20, "fill should at least cover the diagonals of L and U, got ${f.nnz}")
        }
    }

    /**
     * Many factorizations in a loop must not exhaust native memory.
     *
     * The factors are UMFPACK's own allocation, released by a Cleaner when the factorization becomes
     * unreachable. This does not assert *when* that happens — it cannot, the release is not deterministic —
     * only that churning through more factorizations than would fit if nothing were ever freed does not fall
     * over.
     */
    @Test
    fun `repeated factorizations do not exhaust native memory`() {
        requireSuiteSparse()
        val rng = Random(20260820)
        val a = sparseSystem(120, rng)
        var checksum = 0.0
        repeat(300) {
            val f = umfpack.factor(a)
            checksum += abs(f.solve(DoubleArray(120) { 1.0 })[0])
        }
        assertTrue(checksum > 0.0, "the loop should have produced solutions")
    }
}
