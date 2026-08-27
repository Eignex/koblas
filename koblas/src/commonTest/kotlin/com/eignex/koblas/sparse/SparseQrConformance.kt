package com.eignex.koblas.sparse

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.random.Random
import kotlin.test.assertEquals

internal fun sparseQrConformanceSystem(m: Int, n: Int, rng: Random): F64SparseMatrix {
    val columns = ArrayList<List<Pair<Int, Double>>>(n)
    for (j in 0 until n) {
        val rows = (listOf(j) + List(3) { rng.nextInt(m) }).distinct().sorted()
        columns.add(rows.map { row -> row to if (row == j) 2.0 + rng.nextDouble() else rng.nextDouble(-1.0, 1.0) })
    }
    return F64SparseMatrix.ofColumns(m, n, columns)
}

internal fun assertQrAgreesWithReference(decompositions: F64SparseDecompositions) {
    val rng = Random(20260827)
    for ((m, n) in listOf(1 to 1, 9 to 4, 60 to 25, 140 to 90)) {
        val a = sparseQrConformanceSystem(m, n, rng)
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }

        decompositions.qr(a).use { host ->
            assertEquals(m, host.m, "${m}x$n the factorization reports the wrong row count")
            assertEquals(n, host.n, "${m}x$n the factorization reports the wrong column count")
            assertEquals(n, host.rank, "${m}x$n a full-rank system came back deficient")
            val x = host.solve(b)
            F64ReferenceSparseLinearAlgebra.qr(a).use { portable ->
                assertClose(portable.solve(b), x, "${m}x$n", tolerance = 1e-7)
            }
            assertClose(
                transposeTimes(a, b),
                transposeTimes(a, times(a, x)),
                "${m}x$n normal equations",
                tolerance = 1e-7,
            )
            assertUpperTriangularFactor(a, host, "${m}x$n")
            assertFactorizationIdentity(a, host, "${m}x$n")
            assertOrthogonalOperator(host, "${m}x$n", rng)
        }
    }
}

private fun assertFactorizationIdentity(a: F64SparseMatrix, qr: F64SparseQrFactorization, context: String) {
    val order = qr.columnOrder
    for (j in 0 until qr.n) {
        val embedded = DoubleArray(qr.m)
        qr.r.forEachInColumn(j) { row, value -> embedded[row] = value }
        val actual = qr.applyQ(embedded)
        val expected = DoubleArray(qr.m)
        a.forEachInColumn(order[j]) { row, value -> expected[row] = value }
        assertClose(expected, actual, "$context Q R column $j", tolerance = 1e-7)
    }
}

private fun assertUpperTriangularFactor(a: F64SparseMatrix, qr: F64SparseQrFactorization, context: String) {
    val r = qr.r
    val order = qr.columnOrder
    assertEquals(qr.n, r.cols, "$context R has the wrong column count")
    assertEquals(qr.n, order.size, "$context the column ordering has the wrong length")
    for (j in 0 until qr.n) {
        for (i in 0 until qr.n) {
            assertClose(
                columnDot(a, order[i], order[j]),
                columnDot(r, i, j),
                "$context RᵀR($i, $j)",
                tolerance = 1e-6,
            )
        }
    }
}

private fun assertOrthogonalOperator(qr: F64SparseQrFactorization, context: String, rng: Random) {
    val y = DoubleArray(qr.m) { rng.nextDouble(-1.0, 1.0) }

    val roundTrip = qr.applyQ(qr.applyQ(y, transpose = true))

    assertClose(y, roundTrip, "$context Q round trip", tolerance = 1e-8)
}

private fun columnDot(a: F64SparseMatrix, i: Int, j: Int): Double {
    val column = HashMap<Int, Double>()
    a.forEachInColumn(i) { row, value -> column[row] = value }
    var sum = 0.0
    a.forEachInColumn(j) { row, value -> column[row]?.let { sum += it * value } }
    return sum
}

private fun times(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
    val y = DoubleArray(a.rows)
    for (j in 0 until a.cols) a.forEachInColumn(j) { row, value -> y[row] += value * x[j] }
    return y
}

private fun transposeTimes(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
    val y = DoubleArray(a.cols)
    for (j in 0 until a.cols) a.forEachInColumn(j) { row, value -> y[j] += value * x[row] }
    return y
}
