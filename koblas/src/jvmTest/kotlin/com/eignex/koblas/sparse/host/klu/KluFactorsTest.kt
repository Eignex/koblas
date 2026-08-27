package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.assertLuFactorsReproduce
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

@Category(HostLibraryTest::class)
class KluFactorsTest {
    private val klu = KluSparseLu(KluConfig(KluOptions(factorizeMin = 0)))

    private fun requireKlu() {
        Assume.assumeTrue("KLU is not installed", klu.isAvailable)
    }

    @Test
    fun `the extracted factors reproduce the scaled and permuted matrix`() {
        requireKlu()
        val rng = Random(20260827)
        for (n in intArrayOf(4, 16, 64)) {
            val a = dominant(n, rng)

            klu.factor(a).use { factorization ->
                val lu = factorization as F64SparseLuFactorization
                assertEquals(n, lu.rowOrder.size, "n=$n rowOrder")
                assertLuFactorsReproduce(a, lu, "n=$n")
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
