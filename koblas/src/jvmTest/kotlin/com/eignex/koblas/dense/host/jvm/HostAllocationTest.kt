package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * What a destination-passing routine costs on the host backend. An FFM call allocates a memory-segment
 * wrapper per array, so nothing here reaches zero; the point is that reusing a destination must not carry
 * the factor buffer the caller passed one to avoid, which is what routing through a fresh factorization does.
 */
@Category(HostLibraryTest::class)
class HostAllocationTest {

    @Test
    fun `factorInto does not carry the factor buffer a fresh factorization allocates`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostLibraries.lapacke)
        val host = F64Lapacke()
        // Above the LAPACK gate of 64, so the host path runs rather than the portable fallback.
        val n = 96
        val rng = Random(20260824)
        val a = F64DenseMatrix(2 * n, n)
        for (j in 0 until n) {
            for (i in 0 until 2 * n) {
                a[i, j] = if (i == j) 2.0 else rng.nextDouble(-1.0, 1.0)
            }
        }
        val reused = host.factor(a)

        val fresh = bytesPerIteration(200) { host.factor(a) }
        val into = bytesPerIteration(200) { host.factorInto(a, reused) }
        val buffer = 2.0 * n * n * Double.SIZE_BYTES
        assertTrue(fresh > buffer * 0.5, "a fresh factorization should allocate the factor, saw $fresh B")
        assertTrue(
            into < fresh / 10.0,
            "factorInto allocated $into B against $fresh B for a fresh factorization, so it is still " +
                "routing through one instead of writing into the destination",
        )
    }
}
