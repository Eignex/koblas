package com.eignex.koblas.sparse.factorization.qr

import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class F64SparseHouseholderQrTest {
    @Test
    fun `the least squares solution satisfies the normal equations`() {
        val rng = Random(20260827)
        for ((m, n) in listOf(1 to 1, 5 to 3, 40 to 12, 97 to 61)) {
            val a = tallSparseMatrix(m, n, rng)
            val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }

            val x = F64SparseHouseholderQr.factor(a).solve(b)

            assertClose(transposeMultiply(a, b), transposeMultiply(a, multiply(a, x)), "${m}x$n", tolerance = 1e-8)
        }
    }

    @Test
    fun `the residual is orthogonal to every column of the matrix`() {
        val rng = Random(20260828)
        val a = tallSparseMatrix(60, 20, rng)
        val b = DoubleArray(60) { rng.nextDouble(-1.0, 1.0) }

        val residual = multiply(a, F64SparseHouseholderQr.factor(a).solve(b)).mapIndexed { i, v -> v - b[i] }

        assertClose(DoubleArray(20), transposeMultiply(a, residual.toDoubleArray()), "residual", tolerance = 1e-8)
    }

    @Test
    fun `a square system is solved exactly`() {
        val rng = Random(20260829)
        val a = tallSparseMatrix(30, 30, rng)
        val expected = DoubleArray(30) { rng.nextDouble(-1.0, 1.0) }

        val x = F64SparseHouseholderQr.factor(a).solve(multiply(a, expected))

        assertClose(expected, x, "square", tolerance = 1e-7)
    }

    @Test
    fun `R transposed times R reproduces A transposed times A`() {
        val rng = Random(20260830)
        val a = tallSparseMatrix(25, 9, rng)

        val r = F64SparseHouseholderQr.factor(a).r

        for (j in 0 until 9) {
            for (i in 0 until 9) {
                assertClose(
                    columnDot(a, i, j),
                    columnDot(r, i, j),
                    "RᵀR($i, $j)",
                    tolerance = 1e-8,
                )
            }
        }
    }

    @Test
    fun `applying Q transposed and then Q returns the original vector`() {
        val rng = Random(20260831)
        val a = tallSparseMatrix(45, 15, rng)
        val y = DoubleArray(45) { rng.nextDouble(-1.0, 1.0) }
        val qr = F64SparseHouseholderQr.factor(a)

        val roundTrip = qr.applyQInto(
            qr.applyQInto(y, DoubleArray(45), transpose = true),
            DoubleArray(45),
        )

        assertClose(y, roundTrip, "round trip", tolerance = 1e-9)
    }

    @Test
    fun `a rank deficient matrix is reported rather than solved`() {
        // Column 2 repeats column 0, so R(2, 2) is zero and the minimizer is a line rather than a point.
        val a = F64SparseMatrix.ofColumns(
            4,
            3,
            listOf(
                listOf(0 to 1.0, 1 to 2.0),
                listOf(1 to 1.0, 2 to 3.0),
                listOf(0 to 1.0, 1 to 2.0),
            ),
        )

        val qr = F64SparseHouseholderQr.factor(a)

        assertEquals(2, qr.rank)
        assertTrue(qr.rankDeficient)
        assertFailsWith<SingularMatrix> { qr.solve(DoubleArray(4) { 1.0 }) }
    }

    @Test
    fun `a wider than tall matrix is rejected`() {
        val a = F64SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(0 to 1.0)))

        assertFailsWith<IllegalArgumentException> { F64SparseHouseholderQr.factor(a) }
    }

    @Test
    fun `an empty column leaves the factorization rank deficient`() {
        val a = F64SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 2.0), emptyList()))

        val qr = F64SparseHouseholderQr.factor(a)

        assertEquals(1, qr.rank)
        assertEquals(3, qr.m)
        assertEquals(2, qr.n)
    }

    @Test
    fun `large finite values do not overflow the reflector`() {
        val a = F64SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 1e200, 1 to 1e200)))

        val x = F64SparseHouseholderQr.factor(a).solve(doubleArrayOf(1e200, 1e200))

        assertTrue(x[0].isFinite())
        assertEquals(1.0, x[0], 1e-12)
    }

    @Test
    fun `small finite values do not lose the reflector tail`() {
        val a = F64SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 1e-200, 1 to 1e-200)))

        val x = F64SparseHouseholderQr.factor(a).solve(doubleArrayOf(0.0, 1e-200))

        assertEquals(0.5, x[0], 1e-12)
    }
}

private fun tallSparseMatrix(m: Int, n: Int, rng: Random): F64SparseMatrix {
    val columns = ArrayList<List<Pair<Int, Double>>>(n)
    for (j in 0 until n) {
        val rows = (listOf(j) + List(3) { rng.nextInt(m) }).distinct().sorted()
        columns.add(rows.map { row -> row to if (row == j) 2.0 + rng.nextDouble() else rng.nextDouble(-1.0, 1.0) })
    }
    return F64SparseMatrix.ofColumns(m, n, columns)
}

private fun multiply(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
    val y = DoubleArray(a.rows)
    for (j in 0 until a.cols) a.forEachInColumn(j) { row, value -> y[row] += value * x[j] }
    return y
}

private fun transposeMultiply(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
    val y = DoubleArray(a.cols)
    for (j in 0 until a.cols) a.forEachInColumn(j) { row, value -> y[j] += value * x[row] }
    return y
}

private fun columnDot(a: F64SparseMatrix, i: Int, j: Int): Double {
    val column = HashMap<Int, Double>()
    a.forEachInColumn(i) { row, value -> column[row] = value }
    var sum = 0.0
    a.forEachInColumn(j) { row, value -> column[row]?.let { sum += it * value } }
    return if (abs(sum) < 1e-300) 0.0 else sum
}
