package com.eignex.koblas.sparse

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.test.assertEquals

/*
 * The identity each kind of factorization satisfies, checked against the matrix rather than against another
 * implementation. Orderings, signs and scalings are a provider's own choice, so these are written in terms of
 * what it reports: a backend that permuted differently satisfies the same identity with its own permutation.
 */

/** `L·U + F = P·diag(rowScaling)·A·Q`, the identity an LU reports its factors against. */
internal fun assertLuFactorsReproduce(a: F64SparseMatrix, lu: F64SparseLuFactorization, context: String) {
    val n = a.rows
    val position = IntArray(n)
    for (k in 0 until n) position[lu.rowOrder[k]] = k
    val expected = DoubleArray(n * n)
    val columnOrder = lu.columnOrder
    val scaling = lu.rowScaling
    for (k in 0 until n) {
        a.forEachInColumn(columnOrder[k]) { row, value -> expected[position[row] + k * n] = value * scaling[row] }
    }
    assertClose(expected, add(product(lu.l, lu.u), dense(lu.offDiagonal)), "$context L*U + F", tolerance = 1e-8)
}

/** `L·Lᵀ = P·A·Pᵀ`, over the full matrix the stored lower triangle stands for. */
internal fun assertCholeskyFactorReproduces(
    a: F64SparseMatrix,
    cholesky: F64SparseCholeskyFactorization,
    context: String,
) {
    assertClose(
        permuted(symmetrized(a), cholesky.order),
        gram(cholesky.l) { 1.0 },
        "$context L*Lt",
        tolerance = 1e-8,
    )
}

/**
 * `L·D·Lᵀ = P·A·Pᵀ`, over the full matrix the stored lower triangle stands for.
 *
 * `L`'s unit diagonal is implicit, so it is supplied here rather than read; that is the whole difference
 * between this and the Cholesky identity above.
 */
internal fun assertLdlFactorsReproduce(a: F64SparseMatrix, ldl: F64QuasiDefiniteLdlFactorization, context: String) {
    val d = ldl.d
    assertEquals(
        FactorizationInertia(
            positive = d.count { it > 0.0 },
            negative = d.count { it < 0.0 },
            zero = d.count { it == 0.0 },
        ),
        ldl.inertia,
        "$context inertia",
    )
    assertClose(
        permuted(symmetrized(a), ldl.order),
        gram(withUnitDiagonal(ldl.l)) { d[it] },
        "$context L*D*Lt",
        tolerance = 1e-8,
    )
}

/** [l] with the unit diagonal an `L·D·Lᵀ` keeps implicit put back, so one product formula serves both kinds. */
private fun withUnitDiagonal(l: F64SparseMatrix): F64SparseMatrix {
    val n = l.rows
    val colPtr = IntArray(n + 1)
    val columns = List(n) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        entries.add(j to 1.0)
        l.forEachInColumn(j) { i, v -> entries.add(i to v) }
        entries
    }
    for (k in 0 until n) colPtr[k + 1] = colPtr[k] + columns[k].size
    return F64SparseMatrix.ofColumns(n, n, columns)
}

/** The full symmetric matrix a stored lower triangle stands for, dense and column-major. */
private fun symmetrized(a: F64SparseMatrix): DoubleArray {
    val n = a.rows
    val out = DoubleArray(n * n)
    for (j in 0 until n) {
        a.forEachInColumn(j) { i, v ->
            out[i + j * n] = v
            out[j + i * n] = v
        }
    }
    return out
}

private fun permuted(full: DoubleArray, order: IntArray): DoubleArray {
    val n = order.size
    val position = IntArray(n)
    for (k in 0 until n) position[order[k]] = k
    val out = DoubleArray(n * n)
    for (j in 0 until n) {
        for (i in 0 until n) out[position[i] + position[j] * n] = full[i + j * n]
    }
    return out
}

/** `L·diag(scale)·Lᵀ`, dense and column-major. */
private inline fun gram(l: F64SparseMatrix, scale: (Int) -> Double): DoubleArray {
    val n = l.rows
    val out = DoubleArray(n * n)
    for (k in 0 until n) {
        val factor = scale(k)
        l.forEachInColumn(k) { i, lik ->
            l.forEachInColumn(k) { j, ljk -> out[i + j * n] += lik * factor * ljk }
        }
    }
    return out
}

private fun product(l: F64SparseMatrix, u: F64SparseMatrix): DoubleArray {
    val n = l.rows
    val out = DoubleArray(n * n)
    for (j in 0 until n) {
        u.forEachInColumn(j) { k, ukj -> l.forEachInColumn(k) { i, lik -> out[i + j * n] += lik * ukj } }
    }
    return out
}

private fun dense(a: F64SparseMatrix): DoubleArray {
    val out = DoubleArray(a.rows * a.cols)
    for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> out[i + j * a.rows] = v }
    return out
}

private fun add(a: DoubleArray, b: DoubleArray): DoubleArray = DoubleArray(a.size) { a[it] + b[it] }
