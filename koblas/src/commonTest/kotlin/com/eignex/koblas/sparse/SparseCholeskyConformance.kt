package com.eignex.koblas.sparse

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * A symmetric positive-definite system as the lower triangle a Cholesky reads, diagonally dominant so it
 * factors without an ordering. The off-diagonal pattern is symmetric by construction, which is what makes
 * the stored triangle describe a symmetric matrix at all.
 */
internal fun sparseSymmetricConformanceSystem(n: Int, rng: Random): F64SparseMatrix {
    val below = Array(n) { HashMap<Int, Double>() }
    val weight = DoubleArray(n)
    for (j in 0 until n) {
        for (i in j + 1 until n) {
            if (rng.nextDouble() >= 0.25) continue
            val v = rng.nextDouble(-1.0, 1.0)
            below[j][i] = v
            weight[i] += abs(v)
            weight[j] += abs(v)
        }
    }
    val columns = ArrayList<List<Pair<Int, Double>>>(n)
    for (j in 0 until n) {
        val column = ArrayList<Pair<Int, Double>>()
        column.add(j to weight[j] + 1.0)
        for (i in j + 1 until n) below[j][i]?.let { column.add(i to it) }
        columns.add(column)
    }
    return F64SparseMatrix.ofColumns(n, n, columns)
}

/**
 * The Cholesky half of a backend against the portable one. A backend with no native Cholesky answers through
 * the adapter's fallback, which is what this checks for it; one that has its own is checked against the
 * definition the same way its factorization already is.
 */
internal fun assertCholeskyAgreesWithReference(decompositions: F64SparseDecompositions) {
    val rng = Random(20260911)
    for (n in intArrayOf(1, 2, 7, 23)) {
        val a = sparseSymmetricConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val host = decompositions.cholesky(a)
        val portable = F64ReferenceSparseLinearAlgebra.cholesky(a)
        assertEquals(n, host.n, "n=$n the factorization reports the wrong dimension")
        assertTrue(!host.singular, "n=$n a positive-definite system came back singular")
        assertClose(portable.solve(b), host.solve(b), "n=$n", tolerance = 1e-9)
        assertClose(b, multiply(a.symmetrized(), host.solve(b)), "n=$n residual", tolerance = 1e-9)
    }
}

/** The full symmetric matrix a stored lower triangle stands for, so a residual can be taken against it. */
private fun F64SparseMatrix.symmetrized(): F64SparseMatrix {
    val columns = List(cols) { ArrayList<Pair<Int, Double>>() }
    for (j in 0 until cols) {
        forEachInColumn(j) { i, v ->
            columns[j].add(i to v)
            if (i != j) columns[i].add(j to v)
        }
    }
    return F64SparseMatrix.ofColumns(rows, cols, columns.map { it.sortedBy(Pair<Int, Double>::first) })
}

/**
 * The `L·D·Lᵀ` half of a backend against the portable one, on a matrix that is indefinite, since agreeing on
 * a positive definite one would not tell this factorization apart from a Cholesky.
 */
internal fun assertLdlAgreesWithReference(decompositions: F64SparseDecompositions) {
    val rng = Random(20260927)
    for (n in intArrayOf(2, 8, 21)) {
        val a = indefiniteConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val host = decompositions.quasiDefiniteLdl(a)
        val portable = F64ReferenceSparseLinearAlgebra.quasiDefiniteLdl(a)
        assertTrue(!host.singular, "n=$n an invertible system came back singular")
        assertClose(b, multiply(a.symmetrized(), host.solve(b)), "n=$n residual", tolerance = 1e-8)
        assertClose(portable.solve(b), host.solve(b), "n=$n", tolerance = 1e-8)
    }
}

/**
 * A symmetric indefinite system as its stored lower triangle: diagonally dominant in magnitude, so it is
 * invertible and its factorization is well behaved, with the sign of the diagonal alternating so it is not
 * positive definite and a Cholesky would refuse it.
 */
internal fun indefiniteConformanceSystem(n: Int, rng: Random): F64SparseMatrix {
    val below = Array(n) { HashMap<Int, Double>() }
    val weight = DoubleArray(n)
    for (j in 0 until n) {
        for (i in j + 1 until n) {
            if (rng.nextDouble() >= 0.25) continue
            val v = rng.nextDouble(-1.0, 1.0)
            below[j][i] = v
            weight[i] += abs(v)
            weight[j] += abs(v)
        }
    }
    val columns = ArrayList<List<Pair<Int, Double>>>(n)
    for (j in 0 until n) {
        val column = ArrayList<Pair<Int, Double>>()
        column.add(j to if (j % 2 == 0) weight[j] + 1.0 else -(weight[j] + 1.0))
        for (i in j + 1 until n) below[j][i]?.let { column.add(i to it) }
        columns.add(column)
    }
    return F64SparseMatrix.ofColumns(n, n, columns)
}

/**
 * The Cholesky analysis of a provider against its own ordinary factorization. Every provider owes the same
 * two promises whether or not it holds a symbolic pass of its own: the same factor a fresh call gives, and a
 * refusal of a pattern it did not analyze.
 */
internal fun assertCholeskyAnalysisReuses(provider: F64SparseCholesky) {
    val a = sparseSymmetricConformanceSystem(21, Random(20260935))
    val other = sparseSymmetricConformanceSystem(21, Random(20260936))
    val b = DoubleArray(21) { it * 0.125 - 1.0 }
    val expected = provider.cholesky(a).use { it.solve(b) }

    provider.analyzeCholesky(a).use { analysis ->
        analysis.factor(a).use { assertClose(expected, it.solve(b), "analyzed solve", tolerance = 1e-12) }
        assertFailsWith<IllegalArgumentException> { analysis.factor(other) }
    }
}
