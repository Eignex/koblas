package com.eignex.koblas.dense

import com.eignex.koblas.testutil.allocation.bytesPerIteration
import java.util.concurrent.Executors
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ScalarBlockKernelsRuntimeTest {
    @Test
    fun `prebound scalar products allocate no operation scratch`() {
        val a = MatrixWindow(DoubleArray(16) { 0.25 }, 4, 4)
        val b = ScalarLayoutKernels.pack(a, PackedMatrixLayout(PackedRole.Right, 4, 4, 3))
        val c = MatrixWindow(DoubleArray(16), 4, 4)
        val bytes = bytesPerIteration(500, warmup = 1000) { ScalarBlockKernels.product(a, b, c) }
        assertTrue(bytes <= 16.0, "prebound scalar block allocated $bytes bytes")
    }

    @Test
    fun `retained inputs can be shared across independent concurrent outputs`() {
        val a = ScalarLayoutKernels.pack(
            MatrixWindow(DoubleArray(16) { 0.5 }, 4, 4),
            PackedMatrixLayout(PackedRole.Left, 4, 4, 3),
        )
        val b = ScalarLayoutKernels.pack(
            MatrixWindow(DoubleArray(16) { 0.25 }, 4, 4),
            PackedMatrixLayout(PackedRole.Right, 4, 4, 2),
        )
        val pool = Executors.newFixedThreadPool(2)
        try {
            val futures = (1..2).map { multiplier ->
                pool.submit<DoubleArray> {
                    val output = DoubleArray(16)
                    ScalarBlockKernels.product(a, b, MatrixWindow(output, 4, 4), alpha = multiplier.toDouble())
                    output
                }
            }
            for ((index, future) in futures.withIndex()) {
                future.get().forEach { assertEquals((index + 1) * 0.5, it) }
            }
        } finally {
            pool.shutdownNow()
        }
    }
}
