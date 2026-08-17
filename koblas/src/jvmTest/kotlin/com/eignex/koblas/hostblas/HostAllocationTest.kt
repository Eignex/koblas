package com.eignex.koblas.hostblas

import com.eignex.koblas.HostLibraryTest
import com.eignex.koblas.bytesPerIteration
import com.eignex.koblas.wellConditioned
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * What a destination-passing routine costs on the host backend. An FFM call allocates a memory-segment
 * wrapper per array, so nothing here reaches zero; the point is that reusing a destination must not carry
 * the `n²` buffer the caller passed one to avoid, which is what routing through a fresh factorization does.
 */
@Category(HostLibraryTest::class)
class HostAllocationTest {

    @Test
    fun `factorInto does not carry the factor buffer a fresh factorization allocates`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        val host = HostLapack()
        // Above the LAPACK gate of 64, so the host path runs rather than the portable fallback.
        val n = 96
        val rng = Random(20260824)
        val a = wellConditioned(n, rng)
        val reused = host.factor(a)

        val fresh = bytesPerIteration(200) { host.factor(a) }
        val into = bytesPerIteration(200) { host.factorInto(a, reused) }
        val buffer = n.toDouble() * n * Double.SIZE_BYTES
        assertTrue(fresh > buffer * 0.5, "a fresh factorization should allocate the n² factor, saw $fresh B")
        assertTrue(
            into < fresh / 10.0,
            "factorInto allocated $into B against $fresh B for a fresh factorization, so it is still " +
                "routing through one instead of writing into the destination",
        )
    }
}
