package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.randomVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SparseOrderingTest {

    private fun arrow(n: Int): SparseMatrix {
        val columns = List(n) { j ->
            buildList {
                when (j) {
                    0 -> {
                        add(0 to 4.0 * n)
                        for (i in 1 until n) add(i to 1.0)
                    }

                    else -> {
                        add(0 to 1.0)
                        add(j to 4.0)
                    }
                }
            }
        }
        return SparseMatrix.ofColumns(n, n, columns)
    }

    /** A·x from the CSC arrays, so a solve is checked against the matrix rather than a seam. */
    private fun multiply(a: SparseMatrix, x: DoubleArray): DoubleArray {
        val y = DoubleArray(a.rows)
        for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> y[i] += v * x[j] }
        return y
    }

    @Test
    fun `the ordering is a permutation and Natural is the identity`() {
        val a = arrow(24)
        val ordered = a.analyze()
        assertEquals((0 until 24).toList(), ordered.permutation.sorted(), "not a permutation")
        val natural = a.analyze(SparseOrdering.Natural)
        assertEquals((0 until 24).toList(), natural.permutation.toList(), "Natural must eliminate in order")
    }

    @Test
    fun `reordering collapses the fill an arrow matrix would otherwise produce`() {
        val n = 60
        val a = arrow(n)
        val natural = a.analyze(SparseOrdering.Natural).nnz
        val ordered = a.analyze().nnz
        assertEquals((n - 1) * n / 2, natural, "the natural order should fill an arrow completely")
        assertEquals(n - 1, ordered, "the ordering should leave an arrow with no fill at all")
    }

    @Test
    fun `reordering does not change the answer`() {
        val rng = Random(20260814)
        val a = arrow(40)
        val x = randomVector(40, rng)
        val b = multiply(a, x)
        for (ordering in SparseOrdering.entries) {
            val f = a.cholesky(ordering = ordering)
            assertClose(x, f.solve(b), "solve under $ordering", tolerance = 1e-9)
            assertClose(x, f.solve(b, transpose = true), "transposed solve under $ordering", tolerance = 1e-9)
        }
    }

    @Test
    fun `a shuffled band is reordered back to something cheap`() {
        val n = 80
        val rng = Random(20260815)
        val shuffle = (0 until n).shuffled(rng)
        val entries = Array(n) { HashMap<Int, Double>() }
        for (i in 0 until n) entries[shuffle[i]][shuffle[i]] = 8.0
        for (i in 0 until n - 1) {
            entries[shuffle[i]][shuffle[i + 1]] = -1.0
            entries[shuffle[i + 1]][shuffle[i]] = -1.0
        }
        val a = SparseMatrix.ofColumns(n, n, List(n) { j -> entries[j].map { (i, v) -> i to v } })

        val natural = a.analyze(SparseOrdering.Natural).nnz
        val ordered = a.analyze().nnz
        assertTrue(ordered <= n - 1, "a permuted band should come back to band fill, got $ordered")
        assertTrue(ordered < natural, "the ordering should beat the shuffled order ($ordered vs $natural)")

        val x = randomVector(n, rng)
        assertClose(x, a.cholesky().solve(multiply(a, x)), "solve after reordering", tolerance = 1e-9)
    }

    /** The upper triangle of a tridiagonal band, whose minimum-degree ordering leaves no fill at all. */
    private fun band(n: Int): SparseMatrix {
        val rowIdx = IntArray(2 * n - 1)
        val colIdx = IntArray(2 * n - 1)
        val values = DoubleArray(2 * n - 1)
        var k = 0
        for (j in 0 until n) {
            if (j > 0) {
                rowIdx[k] = j - 1
                colIdx[k] = j
                values[k] = -1.0
                k++
            }
            rowIdx[k] = j
            colIdx[k] = j
            values[k] = 4.0
            k++
        }
        return SparseMatrix.ofTriplets(n, n, rowIdx, colIdx, values)
    }

    /**
     * The assertion is on fill, not on the clock, but the size is the point: an ordering that rescores every
     * variable at every step cannot finish at this n, so it fails as a job that runs out of time.
     */
    @Test
    fun `the ordering scales to a band far past what an all-pairs degree scan survives`() {
        val n = 20_000
        assertEquals(n - 1, band(n).analyze().nnz, "a band should order to no fill at all")
    }

    /** A 5-point Laplacian, where the ordering has to do real work rather than peel off leaves. */
    @Test
    fun `a two-dimensional grid orders to a fraction of the natural fill`() {
        val k = 40
        val n = k * k
        val rowIdx = ArrayList<Int>()
        val colIdx = ArrayList<Int>()
        val values = ArrayList<Double>()
        fun entry(i: Int, j: Int, v: Double) {
            rowIdx.add(i)
            colIdx.add(j)
            values.add(v)
        }
        for (y in 0 until k) {
            for (x in 0 until k) {
                val i = y * k + x
                entry(i, i, 8.0)
                if (x + 1 < k) entry(i, i + 1, -1.0)
                if (y + 1 < k) entry(i, i + k, -1.0)
            }
        }
        val a = SparseMatrix.ofTriplets(n, n, rowIdx.toIntArray(), colIdx.toIntArray(), values.toDoubleArray())
        val natural = a.analyze(SparseOrdering.Natural).nnz
        val ordered = a.analyze().nnz
        assertTrue(ordered * 3 < natural, "grid fill $ordered should be well under the natural $natural")
    }

    @Test
    fun `an analysis with an ordering is reusable across values like any other`() {
        val rng = Random(20260816)
        val a = arrow(30)
        val symbolic = a.analyze()
        val scaled = SparseMatrix(
            a.rows,
            a.cols,
            a.colPtr.copyOf(),
            a.rowIdx.copyOf(),
            DoubleArray(a.values.size) { a.values[it] * 2.0 },
        )
        for (matrix in listOf(a, scaled)) {
            val x = randomVector(30, rng)
            val f = symbolic.factorLdl(matrix)
            assertClose(x, f.solve(multiply(matrix, x)), "reused ordered analysis", tolerance = 1e-9)
        }
    }
}
