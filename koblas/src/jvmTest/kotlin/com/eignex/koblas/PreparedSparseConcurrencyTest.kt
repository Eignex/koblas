package com.eignex.koblas

import java.util.concurrent.Callable
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A prepared sparse matrix as several threads use it at once. The snapshot is immutable and the derived
 * transpose is published through a synchronized lazy, so readers with their own destinations and workspaces
 * need no lock between them.
 */
class PreparedSparseConcurrencyTest {
    private companion object {
        const val ORDER = 96
        const val THREADS = 8
    }

    private fun banded(order: Int): SparseMatrix = SparseMatrix.ofColumns(
        order,
        order,
        List(order) { j ->
            listOfNotNull(
                j to (j % 5 + 1.0),
                if (j + 1 < order) j + 1 to -0.5 else null,
            )
        },
    )

    @Test
    fun `concurrent readers of one prepared matrix agree with a single-threaded call`() {
        val source = banded(ORDER)
        val prepared = source.prepare()
        val x = DoubleArray(ORDER) { it * 0.125 }
        val expected = DoubleArray(ORDER).also { prepared.gemvInto(1.0, x, 0.0, it) }
        val expectedTransposed = DoubleArray(ORDER)
        koblas.gemv(1.0, koblas.transpose(source), x, 0.0, expectedTransposed)
        // Through the one-shot call, so the snapshot's own orientation is still unbuilt when the pool starts.
        val expectedProduct = DenseMatrix.zero(ORDER, ORDER)
        koblas.gemm(1.0, source, true, source, false, 0.0, expectedProduct)

        val barrier = CyclicBarrier(THREADS)
        val pool = Executors.newFixedThreadPool(THREADS)
        try {
            val results = pool.invokeAll(
                (0 until THREADS).map { thread ->
                    Callable {
                        val workspace = Workspace()
                        val y = DoubleArray(ORDER)
                        val product = DenseMatrix.zero(ORDER, ORDER)
                        barrier.await(10, TimeUnit.SECONDS)
                        repeat(50) {
                            prepared.gemvInto(1.0, x, 0.0, y)
                            // A transposed product against a second sparse operand, which is the family that
                            // derives the orientation: concurrent first readers must see a fully built one.
                            prepared.gemmInto(1.0, true, source, false, 0.0, product, workspace)
                        }
                        assertContentEquals(expectedProduct.values, product.values, "thread $thread product")
                        thread to y
                    }
                },
            )
            for (result in results) {
                val (thread, y) = result.get(30, TimeUnit.SECONDS)
                assertContentEquals(expected, y, "thread $thread disagreed")
            }
        } finally {
            pool.shutdownNow()
            assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS), "the pool did not stop")
        }

        val transposed = DoubleArray(ORDER).also { prepared.gemvInto(1.0, x, 0.0, it, transpose = true) }
        assertContentEquals(expectedTransposed, transposed, "the shared transpose was left inconsistent")
    }

    @Test
    fun `a source mutated while readers run does not change the snapshot`() {
        val source = banded(ORDER)
        val prepared = source.prepare()
        val x = DoubleArray(ORDER) { 1.0 }
        val expected = DoubleArray(ORDER).also { prepared.gemvInto(1.0, x, 0.0, it) }

        val pool = Executors.newFixedThreadPool(2)
        try {
            val mutation = pool.submit { repeat(1_000) { source.values.fill(Double.NaN) } }
            val reads = pool.submit(
                Callable {
                    val y = DoubleArray(ORDER)
                    repeat(1_000) { prepared.gemvInto(1.0, x, 0.0, y) }
                    y
                },
            )
            mutation.get(30, TimeUnit.SECONDS)
            assertContentEquals(expected, reads.get(30, TimeUnit.SECONDS))
        } finally {
            pool.shutdownNow()
            assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS), "the pool did not stop")
        }
        assertEquals(ORDER, prepared.cols)
    }
}
