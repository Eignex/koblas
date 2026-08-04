package com.eignex.koblas

import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.koblas
import com.eignex.koblas.dense.koblasInfo
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.solve
import com.eignex.koblas.dense.solveSpd
import com.eignex.koblas.sparse.EtaBasis
import com.eignex.koblas.sparse.SparseLu
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The code in README.md, compiled and run.
 *
 * A sample that no longer compiles is a documentation bug that no reader can work around, and renames
 * like `addOuter` to `ger` are exactly what silently breaks them. This is the cheapest guard: if the
 * README drifts from the API, the build fails rather than a reader discovering it.
 */
class ReadmeSamplesTest {

    @Test
    fun `the LU sample solves the system it claims`() {
        val rows = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))
        val a = DenseMatrix.of(rows)
        val x = a.lu().solve(doubleArrayOf(3.0, 5.0))
        assertClose(doubleArrayOf(0.8, 1.4), x, "README LU sample", tolerance = 1e-12)
    }

    @Test
    fun `the Cholesky sample factors and solves`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        val l = a.cholesky() // A = L·Lᵀ
        val xs = solveSpd(l, doubleArrayOf(3.0, 5.0))
        assertClose(doubleArrayOf(0.8, 1.4), xs, "README Cholesky sample", tolerance = 1e-12)
    }

    @Test
    fun `the sparse sample factorizes and solves both directions`() {
        val cols = listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0))
        val s = SparseMatrix.ofColumns(2, 2, cols)
        val lu = assertNotNull(SparseLu.factorize(s))
        val forward = lu.ftran(doubleArrayOf(3.0, 5.0)) // B x = b
        val backward = lu.btran(doubleArrayOf(3.0, 5.0)) // Bᵀ x = b
        assertClose(doubleArrayOf(0.8, 1.4), forward, "README ftran sample", tolerance = 1e-9)
        assertClose(s.gemv(backward, transpose = true), doubleArrayOf(3.0, 5.0), "README btran", 1e-9)
    }

    @Test
    fun `the workspace sample calls exactly what it shows`() {
        val n = 4
        val rng = Random(20260803)
        val a = wellConditioned(n, rng)
        val b = randomVector(n, rng)
        val identity = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
        val basis = EtaBasis.of(assertNotNull(SparseLu.factorize(identity)), n)
        val lu = a.lu()
        val anorm = norm1(a)
        val threshold = 1e-12
        val iterations = 3

        val ws = Workspace().apply { reserve(n, count = 5) }
        val x = DoubleArray(n)
        repeat(iterations) {
            basis.ftranInto(b, x) // no allocation
            if (koblas.rcond(lu, anorm, ws) < threshold) koblas.factorInto(a, lu)
        }
        // The basis is the identity, so the solve returns b itself.
        assertClose(b, x, "README workspace sample", tolerance = 1e-12)
    }

    @Test
    fun `koblasInfo has the shape the sample shows`() {
        // e.g. "backend=cblas, primitives=scalar+host"
        assertTrue(koblasInfo.startsWith("backend="), koblasInfo)
        assertTrue(", primitives=" in koblasInfo, koblasInfo)
    }
}
