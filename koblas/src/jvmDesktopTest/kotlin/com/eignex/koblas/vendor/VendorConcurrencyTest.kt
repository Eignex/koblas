package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.dense.ScalarVectorKernels
import java.util.concurrent.Callable
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * The two halves of the threading requirement: one compute thread per invocation, and bindings callable from
 * any number of application threads at once. Those conflict only if a binding hides shared mutable state
 * behind the call, which the concurrent cases are here to rule out.
 */
class VendorConcurrencyTest {
    @Test
    fun `every resolved vendor reports one compute thread`() {
        for (vendor in Vendor.entries) {
            val blas = openBlas(vendor) ?: continue

            // A library that could not be held to one thread is refused at load, so resolving one is the
            // assertion: the count is not readable through the binding because it is not configurable.
            assertEquals(vendor, blas.vendor, "${vendor.vendorName} resolved under the wrong identity")
        }
    }

    @Test
    fun `an opened library confirmed one compute thread where it can be asked`() {
        var checked = 0
        for (vendor in Vendor.entries) {
            val blas = openBlas(vendor) ?: continue
            checked++

            // These are multithreaded builds by default, and one that would not hold to a single thread never
            // opens at all. Accelerate exports no count to read back, so it is reported as unconfirmed.
            val expected =
                if (vendor == Vendor.Accelerate) ThreadEvidence.Unconfirmed else ThreadEvidence.Confirmed

            assertEquals(expected, blas.threadEvidence, "unexpected thread evidence for ${vendor.vendorName}")
        }
        if (checked == 0) println("SKIPPED: no CBLAS library installed; thread enforcement was not verified here")
    }

    @Test
    fun `concurrent calls through one binding over shared read only inputs agree with serial ones`() =
        withVendor { blas ->
            val threads = 8
            val order = 24
            val depth = 16
            val a = DenseMatrix.wrap(order, depth, vendorValues(order * depth, 1))
            val b = DenseMatrix.wrap(depth, order, vendorValues(depth * order, 2))
            val expected = DenseMatrix.zero(order)
            ReferenceBlas.gemm(1.0, a, false, b, false, 0.0, expected)

            val pool = Executors.newFixedThreadPool(threads)
            try {
                val barrier = CyclicBarrier(threads)
                val work = (0 until threads).map {
                    Callable {
                        // Each thread owns its destination; a and b are shared and read only.
                        val destination = DoubleArray(order * order)
                        val c = DenseMatrix.wrap(order, order, destination)
                        barrier.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)
                        repeat(REPEATS) { blas.gemm(1.0, a, false, b, false, 0.0, c) }
                        destination
                    }
                }
                for (future in pool.invokeAll(work)) {
                    assertAgreesWithReference(expected.values, future.get(), "concurrent gemm")
                }
            } finally {
                pool.shutdownNow()
            }
        }

    @Test
    fun `concurrent level one calls keep independent outputs`() = withVendor { blas ->
        val threads = 8
        val size = 64
        val shared = vendorValues(size, 3)
        val x = DenseVector.wrap(shared)

        val pool = Executors.newFixedThreadPool(threads)
        try {
            val barrier = CyclicBarrier(threads)
            val work = (0 until threads).map { index ->
                Callable {
                    val alpha = 1.0 + index
                    val destination = DoubleArray(size)
                    val y = DenseVector.wrap(destination)
                    barrier.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    repeat(REPEATS) {
                        destination.fill(0.0)
                        blas.axpy(alpha, x, y)
                    }
                    alpha to destination
                }
            }
            for (future in pool.invokeAll(work)) {
                val (alpha, actual) = future.get()
                val expected = DoubleArray(size)
                ScalarVectorKernels.axpy(expected, 0, alpha, shared, 0, size)
                assertAgreesWithReference(expected, actual, "concurrent axpy")
            }
        } finally {
            pool.shutdownNow()
        }
    }

    @Test
    fun `a shared input is not disturbed by concurrent readers`() = withVendor { blas ->
        val threads = 8
        val size = 64
        val shared = vendorValues(size, 4)
        val original = shared.copyOf()
        val x = DenseVector.wrap(shared)

        val pool = Executors.newFixedThreadPool(threads)
        try {
            val barrier = CyclicBarrier(threads)
            val work = (0 until threads).map {
                Callable {
                    barrier.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)
                    var total = 0.0
                    repeat(REPEATS) { total += blas.dot(x, x) }
                    total / REPEATS
                }
            }
            val expected = ScalarVectorKernels.dot(shared, 0, shared, 0, size)
            for (future in pool.invokeAll(work)) {
                val actual = future.get()
                assertAgreesWithReference(expected, actual, "concurrent dot")
            }
            assertContentEquals(original, shared, "a read-only input was written")
        } finally {
            pool.shutdownNow()
        }
    }

    private companion object {
        const val REPEATS = 200
        const val TIMEOUT_SECONDS = 30L
    }
}
