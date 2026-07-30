package com.eignex.koblas.openblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.LinearAlgebra
import com.eignex.koblas.ReferenceLinearAlgebra
import java.lang.management.GarbageCollectorMXBean
import java.lang.management.ManagementFactory
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The FFM bindings pass heap arrays to native code under `Linker.Option.critical`, which pins them for the
 * duration of each call. Pinning blocks relocation, so the open question was whether a loop of native
 * calls delays collection enough to matter — a `dgemm` at large `n` holds its pin for milliseconds.
 *
 * This drives level-3 work while a second thread churns garbage, then compares collector pause behaviour
 * against the same churn over the portable kernels. It is opt-in via KOBLAS_PINNING=1 because it is a
 * timing measurement rather than a correctness check: it prints numbers to be read, and only asserts the
 * loose bound that pinning does not stall collection outright.
 */
class PinningPressureTest {

    private fun gcSnapshot(): Pair<Long, Long> {
        var count = 0L
        var millis = 0L
        for (bean: GarbageCollectorMXBean in ManagementFactory.getGarbageCollectorMXBeans()) {
            if (bean.collectionCount > 0) count += bean.collectionCount
            if (bean.collectionTime > 0) millis += bean.collectionTime
        }
        return count to millis
    }

    private fun measure(label: String, la: LinearAlgebra, n: Int, rounds: Int): Pair<Long, Long> {
        val rng = Random(20260730)
        val a = DenseMatrix(n, n)
        for (i in 0 until n) for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0)
        val b = DenseMatrix(n, n)
        for (i in 0 until n) for (j in 0 until n) b[i, j] = rng.nextDouble(-1.0, 1.0)
        val c = DenseMatrix(n, n)

        // A concurrent allocator, so collection has to happen while calls hold their pins.
        val stop = AtomicBoolean(false)
        val churn = Thread {
            var sink: DoubleArray? = null
            while (!stop.get()) {
                sink = DoubleArray(4096)
                sink[0] = 1.0
            }
            check(sink == null || sink[0] == 1.0)
        }
        churn.isDaemon = true

        repeat(2) { la.gemm(1.0, a, false, b, false, 0.0, c) } // warm up the native path
        val (countBefore, millisBefore) = gcSnapshot()
        val started = System.nanoTime()
        churn.start()
        repeat(rounds) { la.gemm(1.0, a, false, b, false, 0.0, c) }
        stop.set(true)
        churn.join()
        val elapsedMillis = (System.nanoTime() - started) / 1_000_000
        val (countAfter, millisAfter) = gcSnapshot()
        val collections = countAfter - countBefore
        val paused = millisAfter - millisBefore
        println(
            "$label n=$n rounds=$rounds: wall ${elapsedMillis}ms, collections $collections, " +
                "gc ${paused}ms (${if (collections > 0) paused / collections else 0}ms each)",
        )
        return collections to paused
    }

    @Test
    fun `pinned native calls do not stall collection`() {
        if (System.getenv("KOBLAS_PINNING") == null) return
        for (n in intArrayOf(256, 1024)) {
            val rounds = if (n == 256) 2000 else 100
            val (nativeCollections, nativePause) = measure("ffm/pinned", OpenBlasLinearAlgebra(), n, rounds)
            val (portableCollections, portablePause) = measure("portable", ReferenceLinearAlgebra, n, rounds)
            // The churn thread allocates continuously, so collection must still be happening.
            assertTrue(nativeCollections > 0, "no collections observed under pinning at n=$n")
            println(
                "  n=$n: pinned ${nativePause}ms over $nativeCollections collections vs " +
                    "portable ${portablePause}ms over $portableCollections",
            )
        }
    }
}
