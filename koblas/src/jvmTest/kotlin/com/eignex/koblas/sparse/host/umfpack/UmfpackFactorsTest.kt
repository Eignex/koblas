package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.assertLuFactorsReproduce
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
                assertEquals(n, factorization.rowOrder.size, "n=$n rowOrder")
                assertEquals(n, factorization.columnOrder.size, "n=$n columnOrder")
                assertLuFactorsReproduce(a, factorization, "n=$n")
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
