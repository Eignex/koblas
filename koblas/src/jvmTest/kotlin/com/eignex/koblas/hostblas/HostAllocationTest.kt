package com.eignex.koblas.hostblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HostLibraryTest
import com.sun.management.ThreadMXBean
import org.junit.Assume
import org.junit.experimental.categories.Category
import java.lang.management.ManagementFactory
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

    private val bean = ManagementFactory.getThreadMXBean() as ThreadMXBean

    /** Holds each result so escape analysis cannot delete the allocation being measured. */
    private var sink: Any? = null

    private fun bytesPerIteration(iterations: Int, block: () -> Any?): Double {
        repeat(200) { sink = block() }
        val id = Thread.currentThread().threadId()
        var best = Double.MAX_VALUE
        repeat(5) {
            val before = bean.getThreadAllocatedBytes(id)
            repeat(iterations) { sink = block() }
            val after = bean.getThreadAllocatedBytes(id)
            best = minOf(best, (after - before).toDouble() / iterations)
        }
        return best
    }

    @Test
    fun `factorInto does not carry the factor buffer a fresh factorization allocates`() {
        Assume.assumeTrue("host LAPACKE is not installed", HostBlasCalls.lapackAvailable)
        val host = HostLapack()
        // Above the LAPACK gate of 64, so the host path runs rather than the portable fallback.
        val n = 96
        val rng = Random(20260824)
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { rng.nextDouble(-1.0, 1.0) })
        for (i in 0 until n) a[i, i] = a[i, i] + n
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
