package com.eignex.koblas.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.HostLibraryTest
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
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

@Category(HostLibraryTest::class)
class UmfpackConformanceTest {

    private val umfpack = UmfpackSparseLapack()

    private fun requireSuiteSparse() {
        Assume.assumeTrue("SuiteSparse is not installed; umfpack conformance cannot run", UmfpackCalls.available)
    }

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
        assertEquals(HOST_BACKEND_PRIORITY, umfpack.priority, "every koblas host binding registers at one priority")
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
                val residual = a.gemv(fromHost, transpose)
                for (i in 0 until n) assertEquals(b[i], residual[i], 1e-9, "n=$n transpose=$transpose residual $i")
            }
        }
    }

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
            // The comparison is relative because the values grow with n.
            assertEquals(1.0, host / portable, 1e-9, "n=$n determinant disagreed: $host vs $portable")
        }
    }

    @Test
    fun `a singular matrix is reported singular with an unknown position`() {
        requireSuiteSparse()
        val rank1 = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to 2.0), listOf(0 to 2.0, 1 to 4.0)),
        )
        val f = umfpack.factor(rank1)
        assertTrue(f.singular, "umfpack should have called a rank-1 matrix singular")
        assertEquals(SINGULAR_POSITION_UNKNOWN, f.failedAt, "a host that cannot name the pivot must say so")
        assertEquals(0.0, f.determinant(), "a singular factorization has determinant zero")
        assertFailsWith<SingularMatrix> { f.solve(doubleArrayOf(1.0, 1.0)) }
    }

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
            assertEquals("reference", koblas.sparseBlas.name)

            val rng = Random(20260819)
            val a = sparseSystem(20, rng)
            val f = koblas.factor(a)
            assertTrue(f.nnz >= 20, "fill should at least cover the diagonals of L and U, got ${f.nnz}")
        }
    }

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

    @Test
    fun `solves run without iterative refinement`() {
        requireSuiteSparse()
        assertEquals(0.0, UmfpackCalls.refinementSteps, "umfpack solves must not refine")
    }

    @Test
    fun `the rest of the control array keeps UMFPACK's defaults`() {
        requireSuiteSparse()
        assertEquals(0.1, UmfpackCalls.pivotTolerance, "umfpack_di_defaults did not fill the control array")
    }
}
