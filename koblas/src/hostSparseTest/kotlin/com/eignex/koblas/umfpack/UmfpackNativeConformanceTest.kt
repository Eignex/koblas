// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.gemv
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class UmfpackNativeConformanceTest {

    /**
     * Required rather than skipped. Kotlin/Native has no Assume, and silently passing a test that exercised
     * nothing is worse than a failure with a reason on it.
     */
    private val umfpack: UmfpackSparseLapack = requireNotNull(
        UmfpackLoader.functions?.let { UmfpackSparseLapack(it) },
    ) { "host SuiteSparse expected in the test environment" }

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
    fun `the binding resolves and registers`() {
        assertEquals("umfpack", umfpack.name)
        assertEquals(HOST_BACKEND_PRIORITY, umfpack.priority, "every koblas host binding registers at one priority")
        assertEquals("umfpack", koblas.sparseLapack.name, "discovery should have registered the backend")
    }

    @Test
    fun `solutions agree with the portable factorization in both directions`() {
        val rng = Random(20260815)
        for (n in intArrayOf(1, 2, 7, 23, 60)) {
            val a = sparseSystem(n, rng)
            val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

            val host = umfpack.factor(a)
            val portable = ReferenceSparseLinearAlgebra.factor(a)
            assertTrue(!host.singular, "n=$n umfpack called a well-conditioned system singular")

            for (transpose in booleanArrayOf(false, true)) {
                val fromHost = host.solve(b, transpose)
                val fromPortable = portable.solve(b, transpose)
                for (i in 0 until n) {
                    assertTrue(
                        abs(fromPortable[i] - fromHost[i]) < 1e-9,
                        "n=$n transpose=$transpose entry $i: ${fromHost[i]} vs ${fromPortable[i]}",
                    )
                }
                val residual = a.gemv(fromHost, transpose)
                for (i in 0 until n) {
                    assertTrue(abs(b[i] - residual[i]) < 1e-9, "n=$n transpose=$transpose residual $i")
                }
            }
        }
    }

    @Test
    fun `an aliased destination still solves correctly`() {
        val rng = Random(20260816)
        val n = 12
        val a = sparseSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val f = umfpack.factor(a)
        val expected = f.solve(b)
        val aliased = b.copyOf()
        f.solveInto(aliased, aliased)
        for (i in 0 until n) assertTrue(abs(expected[i] - aliased[i]) < 1e-12, "entry $i")
    }

    @Test
    fun `the determinant agrees with the portable factorization`() {
        val rng = Random(20260817)
        for (n in intArrayOf(1, 3, 8)) {
            val a = sparseSystem(n, rng)
            val host = umfpack.factor(a).determinant()
            val portable = ReferenceSparseLinearAlgebra.factor(a).determinant()
            assertTrue(abs(host / portable - 1.0) < 1e-9, "n=$n determinant disagreed: $host vs $portable")
        }
    }

    @Test
    fun `a singular matrix is reported singular with an unknown position`() {
        val rank1 = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to 2.0), listOf(0 to 2.0, 1 to 4.0)),
        )
        val f = umfpack.factor(rank1)
        assertTrue(f.singular, "umfpack should have called a rank-1 matrix singular")
        assertEquals(SINGULAR_POSITION_UNKNOWN, f.failedAt, "a host that cannot name the pivot must say so")
        assertEquals(0, f.nnz, "a singular factorization has no fill")
        assertEquals(0.0, f.determinant(), "a singular factorization has determinant zero")
        assertFailsWith<SingularMatrix> { f.solve(doubleArrayOf(1.0, 1.0)) }
    }

    @Test
    fun `equilibrate and a drop tolerance fall back to the portable factorization`() {
        val rng = Random(20260818)
        val a = sparseSystem(6, rng)
        val b = DoubleArray(6) { rng.nextDouble(-1.0, 1.0) }
        for (fallback in listOf(umfpack.factor(a, equilibrate = true), umfpack.factor(a, dropTolerance = 1e-12))) {
            assertTrue(fallback !is UmfpackFactorization, "a request umfpack cannot serve must not be ignored")
            val residual = a.gemv(fallback.solve(b))
            for (i in 0 until 6) assertTrue(abs(b[i] - residual[i]) < 1e-9, "entry $i")
        }
    }

    /** `usePinned` has no address for an empty array, so these shapes take the portable path instead of UMFPACK. */
    @Test
    fun `empty and all-zero matrices take the portable path`() {
        val empty = umfpack.factor(SparseMatrix.ofColumns(0, 0, emptyList()))
        assertEquals(0, empty.n)
        val zeros = umfpack.factor(SparseMatrix.ofColumns(3, 3, listOf(emptyList(), emptyList(), emptyList())))
        assertTrue(zeros.singular, "a matrix of zeros is singular")
    }

    @Test
    fun `repeated factorizations do not exhaust native memory`() {
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
