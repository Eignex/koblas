package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertNotNull

@Category(HostLibraryTest::class)
class CholmodFactorsTest {
    private val cholmod = CholmodCholesky()

    private fun requireCholmod() {
        Assume.assumeTrue("CHOLMOD is not installed", cholmod.isAvailable)
    }

    @Test
    fun `the Cholesky factor reproduces the permuted matrix`() {
        requireCholmod()
        val rng = Random(20260827)
        for (n in intArrayOf(4, 20, 80)) {
            val a = spdLowerTriangle(n, rng)

            assertNotNull(cholmod.factor(a)).use { factorization ->
                val l = factorization.l
                val order = factorization.order
                assertClose(
                    permuted(a, order),
                    gram(l) { _, _ -> 1.0 },
                    "n=$n L*Lt",
                    tolerance = 1e-8,
                )
            }
        }
    }

    @Test
    fun `the LDL factors reproduce the permuted matrix`() {
        requireCholmod()
        val rng = Random(20260901)
        for (n in intArrayOf(4, 20, 80)) {
            val a = spdLowerTriangle(n, rng)

            assertNotNull(cholmod.factorLdl(a)).use { factorization ->
                val l = factorization.l
                val d = factorization.d
                assertClose(
                    permuted(a, factorization.order),
                    gram(l) { k, _ -> d[k] },
                    "n=$n L*D*Lt",
                    tolerance = 1e-8,
                )
            }
        }
    }
}

private fun spdLowerTriangle(n: Int, rng: Random): F64SparseMatrix {
    val below = List(n) { HashMap<Int, Double>() }
    val weight = DoubleArray(n)
    for (j in 0 until n) {
        for (i in j + 1 until n) {
            if (rng.nextDouble() >= 0.2) continue
            val v = rng.nextDouble(-1.0, 1.0)
            below[j][i] = v
            weight[i] += abs(v)
            weight[j] += abs(v)
        }
    }
    return F64SparseMatrix.ofColumns(
        n,
        n,
        List(n) { j ->
            val column = ArrayList<Pair<Int, Double>>()
            column.add(j to weight[j] + 1.0)
            for (i in j + 1 until n) below[j][i]?.let { column.add(i to it) }
            column
        },
    )
}

/** The full symmetric matrix a stored lower triangle stands for, permuted by [order], column-major dense. */
private fun permuted(a: F64SparseMatrix, order: IntArray): DoubleArray {
    val n = a.rows
    val full = DoubleArray(n * n)
    for (j in 0 until n) {
        a.forEachInColumn(j) { i, v ->
            full[i + j * n] = v
            full[j + i * n] = v
        }
    }
    val position = IntArray(n)
    for (k in 0 until n) position[order[k]] = k
    val out = DoubleArray(n * n)
    for (j in 0 until n) {
        for (i in 0 until n) out[position[i] + position[j] * n] = full[i + j * n]
    }
    return out
}

/** `L·diag(scale)·Lᵀ` as a dense column-major array. */
private fun gram(l: F64SparseMatrix, scale: (Int, Int) -> Double): DoubleArray {
    val n = l.rows
    val out = DoubleArray(n * n)
    for (k in 0 until n) {
        val factor = scale(k, k)
        l.forEachInColumn(k) { i, lik ->
            l.forEachInColumn(k) { j, ljk -> out[i + j * n] += lik * factor * ljk }
        }
    }
    return out
}
