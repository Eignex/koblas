package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.assertCholeskyFactorReproduces
import com.eignex.koblas.sparse.assertLdlFactorsReproduce
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
                assertCholeskyFactorReproduces(a, factorization, "n=$n")
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
                assertLdlFactorsReproduce(a, factorization, "n=$n")
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
