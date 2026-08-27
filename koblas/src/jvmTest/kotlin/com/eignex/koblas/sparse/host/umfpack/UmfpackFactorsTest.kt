package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

@Category(HostLibraryTest::class)
class UmfpackFactorsTest {
    private val umfpack = UmfpackSparseLu(UmfpackConfig(UmfpackOptions(factorizeMin = 0)))

    private fun requireUmfpack() {
        Assume.assumeTrue("UMFPACK is not installed", umfpack.isAvailable)
    }

    @Test
    fun `the extracted factors reproduce the scaled and permuted matrix`() {
        requireUmfpack()
        val rng = Random(20260827)
        for (n in intArrayOf(3, 12, 60)) {
            val a = dominant(n, rng)

            umfpack.factor(a).use { factorization ->
                val lu = factorization as F64SparseLuFactorization
                assertEquals(n, lu.rowOrder.size, "n=$n rowOrder")
                assertEquals(n, lu.columnOrder.size, "n=$n columnOrder")
                assertClose(
                    permutedAndScaled(a, lu.rowOrder, lu.columnOrder, lu.rowScaling),
                    product(lu.l, lu.u),
                    "n=$n L*U",
                    tolerance = 1e-9,
                )
            }
        }
    }
}

private fun dominant(n: Int, rng: Random): F64SparseMatrix {
    val columns = List(n) { j ->
        val entries = ArrayList<Pair<Int, Double>>()
        for (i in 0 until n) {
            when {
                i == j -> entries.add(i to (n + rng.nextDouble()))
                rng.nextDouble() < 0.15 -> entries.add(i to rng.nextDouble(-1.0, 1.0))
            }
        }
        entries
    }
    return F64SparseMatrix.ofColumns(n, n, columns)
}

/** `P·diag(scaling)·A·Q` as a dense column-major array, the right-hand side of the LU identity. */
private fun permutedAndScaled(
    a: F64SparseMatrix,
    rowOrder: IntArray,
    columnOrder: IntArray,
    scaling: DoubleArray,
): DoubleArray {
    val n = a.rows
    val rowAt = IntArray(n)
    for (position in 0 until n) rowAt[rowOrder[position]] = position
    val out = DoubleArray(n * n)
    for (position in 0 until n) {
        val source = columnOrder[position]
        a.forEachInColumn(source) { row, value ->
            out[rowAt[row] + position * n] = value * scaling[row]
        }
    }
    return out
}

private fun product(l: F64SparseMatrix, u: F64SparseMatrix): DoubleArray {
    val n = l.rows
    val out = DoubleArray(n * n)
    for (j in 0 until n) {
        u.forEachInColumn(j) { k, ukj ->
            l.forEachInColumn(k) { i, lik -> out[i + j * n] += lik * ukj }
        }
    }
    return out
}
