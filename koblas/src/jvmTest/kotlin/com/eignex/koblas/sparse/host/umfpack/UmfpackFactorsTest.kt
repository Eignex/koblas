package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.sparse.assertLuFactorsReproduce
import com.eignex.koblas.sparse.sparseDominantSystem
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

@Category(HostLibraryTest::class)
class UmfpackFactorsTest {
    private val umfpack = UmfpackSparseLu(UmfpackConfig(UmfpackOptions()))

    private fun requireUmfpack() {
        Assume.assumeTrue("UMFPACK is not installed", umfpack.isAvailable)
    }

    @Test
    fun `the extracted factors reproduce the scaled and permuted matrix`() {
        requireUmfpack()
        val rng = Random(20260827)
        for (n in intArrayOf(3, 12, 60)) {
            val a = sparseDominantSystem(n, rng)

            umfpack.factor(a).use { factorization ->
                assertEquals(n, factorization.rowOrder.size, "n=$n rowOrder")
                assertEquals(n, factorization.columnOrder.size, "n=$n columnOrder")
                assertLuFactorsReproduce(a, factorization, "n=$n")
            }
        }
    }
}
