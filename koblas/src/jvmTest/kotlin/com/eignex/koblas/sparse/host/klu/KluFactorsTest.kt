package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.core.F64SparseMatrix
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
                assertEquals(n, factorization.rowOrder.size, "n=$n rowOrder")
                assertLuFactorsReproduce(a, factorization, "n=$n")
            }
        }
    }

    @Test
    fun `the extracted factors follow an in place refactorization`() {
        requireKlu()
        val first = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))
        val second = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 5.0)))
        val factorization = klu.factor(first)
        factorization.l

        val refactored = klu.refactor(factorization, second)

        assertLuFactorsReproduce(second, refactored, "refactorized")
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
