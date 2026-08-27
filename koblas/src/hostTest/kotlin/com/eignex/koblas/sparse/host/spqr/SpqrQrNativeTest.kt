// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import kotlin.random.Random
import kotlin.test.*

/** Checks the Kotlin/Native SPQR binding against the portable factorization. */
class SpqrQrNativeTest {
    private val spqr = SpqrQr().also {
        require(it.isAvailable) { "host SPQR expected in the test environment: ${it.unavailableReason}" }
    }

    @Test
    fun `the least squares solution matches the portable factorization`() {
        val rng = Random(20260827)
        for (shape in listOf(1 to 1, 9 to 4, 60 to 25, 400 to 180)) {
            val (m, n) = shape
            val a = tall(m, n, rng)
            val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }

            val native = assertNotNull(spqr.factor(a), "SPQR reported available but produced no factor")
            native.use {
                assertEquals(m, it.m)
                assertEquals(n, it.n)
                assertClose(F64ReferenceSparseLinearAlgebra.qr(a).solve(b), it.solve(b), "${m}x$n", tolerance = 1e-7)
            }
        }
    }

    @Test
    fun `the solution satisfies the normal equations`() {
        val rng = Random(20260901)
        val a = tall(120, 45, rng)
        val b = DoubleArray(120) { rng.nextDouble(-1.0, 1.0) }

        val x = assertNotNull(spqr.factor(a)).use { it.solve(b) }

        assertClose(transposeTimes(a, b), transposeTimes(a, times(a, x)), "normal equations", tolerance = 1e-7)
    }

    @Test
    fun `the explicit factors reproduce the normal equations under the reported ordering`() {
        val rng = Random(20260905)
        val a = tall(50, 18, rng)

        assertNotNull(spqr.factor(a)).use { qr ->
            val r = qr.r
            val order = qr.columnOrder
            assertEquals(18, r.cols)
            assertEquals(18, order.size)
            assertEquals(18, qr.rank)
            assertTrue(qr.nnz > 0, "SPQR reported no stored entries in R")
            for (j in 0 until 18) {
                for (i in 0 until 18) {
                    assertClose(columnDot(a, order[i], order[j]), columnDot(r, i, j), "RtR($i,$j)", tolerance = 1e-6)
                }
            }
        }
    }

    @Test
    fun `applying Q transposed and then Q returns the original vector`() {
        val rng = Random(20260906)
        val a = tall(40, 14, rng)
        val y = DoubleArray(40) { rng.nextDouble(-1.0, 1.0) }

        assertNotNull(spqr.factor(a)).use { qr ->
            assertClose(y, qr.applyQ(qr.applyQ(y, transpose = true)), "Q round trip", tolerance = 1e-8)
        }
    }

    @Test
    fun `a rank deficient matrix is reported as such`() {
        // Column 2 repeats column 0, so SPQR's tolerance should find a rank of two.
        val a = F64SparseMatrix.ofColumns(
            4,
            3,
            listOf(listOf(0 to 1.0, 1 to 2.0), listOf(1 to 1.0, 2 to 3.0), listOf(0 to 1.0, 1 to 2.0)),
        )

        assertNotNull(spqr.factor(a)).use { qr ->
            assertEquals(2, qr.rank)
            assertTrue(qr.rankDeficient)
            assertFailsWith<SingularMatrix> { qr.solve(DoubleArray(4) { 1.0 }) }
        }
    }

    @Test
    fun `a closed factorization refuses to solve and closes idempotently`() {
        val a = tall(20, 8, Random(20260903))
        val factorization = assertNotNull(spqr.factor(a))

        factorization.close()
        factorization.close()

        assertFailsWith<IllegalStateException> { factorization.solve(DoubleArray(20)) }
    }

    @Test
    fun `repeated solves against one factorization agree`() {
        val rng = Random(20260904)
        val a = tall(90, 35, rng)
        val b = DoubleArray(90) { rng.nextDouble(-1.0, 1.0) }

        assertNotNull(spqr.factor(a)).use { factorization ->
            val first = factorization.solve(b)
            repeat(3) { assertClose(first, factorization.solve(b), "repeat", tolerance = 1e-12) }
        }
    }

    @Test
    fun `a wider than tall matrix is rejected before the library is reached`() {
        val a = F64SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(0 to 1.0)))

        assertFailsWith<IllegalArgumentException> { spqr.factor(a) }
    }
}

private fun tall(m: Int, n: Int, rng: Random): F64SparseMatrix {
    val columns = ArrayList<List<Pair<Int, Double>>>(n)
    for (j in 0 until n) {
        val rows = (listOf(j) + List(3) { rng.nextInt(m) }).distinct().sorted()
        columns.add(rows.map { row -> row to if (row == j) 2.0 + rng.nextDouble() else rng.nextDouble(-1.0, 1.0) })
    }
    return F64SparseMatrix.ofColumns(m, n, columns)
}

private fun times(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
    val y = DoubleArray(a.rows)
    for (j in 0 until a.cols) a.forEachInColumn(j) { row, value -> y[row] += value * x[j] }
    return y
}

private fun columnDot(a: F64SparseMatrix, i: Int, j: Int): Double {
    val column = HashMap<Int, Double>()
    a.forEachInColumn(i) { row, value -> column[row] = value }
    var sum = 0.0
    a.forEachInColumn(j) { row, value -> column[row]?.let { sum += it * value } }
    return sum
}

private fun transposeTimes(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
    val y = DoubleArray(a.cols)
    for (j in 0 until a.cols) a.forEachInColumn(j) { row, value -> y[j] += value * x[row] }
    return y
}
