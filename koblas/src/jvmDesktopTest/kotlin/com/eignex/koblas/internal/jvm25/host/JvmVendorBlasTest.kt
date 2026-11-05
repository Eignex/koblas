package com.eignex.koblas.internal.jvm25.host

import com.eignex.koblas.*
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.host.vendorValues
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import com.eignex.koblas.internal.operations.absoluteSumVector
import com.eignex.koblas.internal.operations.axpyVector
import com.eignex.koblas.internal.operations.dotVector
import com.eignex.koblas.internal.operations.normVector
import java.nio.file.Path
import java.util.concurrent.Callable
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The JVM bindings to an installed vendor where they differ from the shared CBLAS conformance: operands staged
 * into foreign memory at strides far apart, shapes the shared cases do not reach, the identity of what was
 * loaded, and the threading requirement. That requirement has two halves, one compute thread per invocation
 * and bindings callable from any number of application threads at once, which conflict only if a binding hides
 * shared mutable state behind the call.
 */
class JvmVendorBlasTest {
    private fun matrix(rows: Int, cols: Int, seed: Int) = DenseMatrix.wrap(rows, cols, vendorValues(rows * cols, seed))

    private fun vector(size: Int, seed: Int) = DenseVector.wrap(vendorValues(size, seed))

    @Test
    fun `an empty operation leaves the destination alone`() = withVendor { blas ->
        val a = DenseMatrix.wrap(3, 0, DoubleArray(0))
        val b = DenseMatrix.wrap(0, 5, DoubleArray(0))
        val storage = vendorValues(15, 13)
        val original = storage.copyOf()

        blas.gemm(1.0, a, false, b, false, 1.0, DenseMatrix.wrap(3, 5, storage))

        assertContentEquals(original, storage)
    }

    @Test
    fun `gemv agrees with the reference under either transpose`() = withVendor { blas ->
        for (transpose in listOf(false, true)) {
            val a = matrix(4, 3, 14)
            val x = vector(if (transpose) 4 else 3, 15)
            val y = vector(if (transpose) 3 else 4, 16)
            val expected = y.values.copyOf()
            ReferenceBlas.gemv(0.5, a, x.values, 2.0, expected, transpose)

            blas.gemv(0.5, a, x.values, x.offset, x.stride, 2.0, y.values, y.offset, y.stride, transpose)

            assertAgreesWithReference(expected, y.values, "gemv transA=$transpose")
        }
    }

    @Test
    fun `a symmetric operand reads only its stored triangle`() = withVendor { blas ->
        val order = 5
        val stored = vendorValues(order * order, 17)
        for (column in 0 until order) {
            for (row in 0 until column) stored[row + column * order] = Double.NaN
        }
        val a = DenseMatrix.wrap(order, order, stored)
        val x = vector(order, 18)
        val y = DenseVector.zero(order)
        val expected = DoubleArray(order)
        ReferenceBlas.symv(1.0, a, x.values, 0.0, expected)

        blas.symv(1.0, a, x.values, x.offset, x.stride, 0.0, y.values, y.offset, y.stride, true)

        assertAgreesWithReference(expected, y.values, "symv")
    }

    @Test
    fun `dot preserves independently strided operand directions`() = withVendor { blas ->
        val size = 4
        for (xStride in listOf(4099, -4099)) {
            for (yStride in listOf(8191, -8191)) {
                val xStorage = vendorValues(3 * abs(xStride) + 5, 37)
                val yStorage = vendorValues(3 * abs(yStride) + 7, 38)
                val xOffset = if (xStride > 0) 2 else 2 + 3 * -xStride
                val yOffset = if (yStride > 0) 3 else 3 + 3 * -yStride
                val x = StridedVector(xStorage, xOffset, size, xStride)
                val y = StridedVector(yStorage, yOffset, size, yStride)
                val expected = ScalarEngine.dot(xStorage, xOffset, yStorage, yOffset, size, xStride, yStride)

                val actual = blas.dotVector(x, y)

                assertAgreesWithReference(expected, actual, "dot strides $xStride and $yStride")
            }
        }
    }

    @Test
    fun `axpy scatters into an independently strided destination`() = withVendor { blas ->
        val size = 4
        for (xStride in listOf(4099, -4099)) {
            for (yStride in listOf(8191, -8191)) {
                val xStorage = vendorValues(3 * abs(xStride) + 5, 39)
                val yStorage = vendorValues(3 * abs(yStride) + 7, 40)
                val expected = yStorage.copyOf()
                val xOffset = if (xStride > 0) 2 else 2 + 3 * -xStride
                val yOffset = if (yStride > 0) 3 else 3 + 3 * -yStride
                val x = StridedVector(xStorage, xOffset, size, xStride)
                val y = StridedVector(yStorage, yOffset, size, yStride)
                ScalarEngine.axpy(
                    expected,
                    yOffset,
                    0.5,
                    xStorage,
                    xOffset,
                    size,
                    yStride = yStride,
                    xStride = xStride,
                )

                blas.axpyVector(0.5, x, y)

                assertAgreesWithReference(expected, yStorage, "axpy backing buffer at strides $xStride and $yStride")
            }
        }
    }

    @Test
    fun `an empty vector call returns the identity without touching storage`() = withVendor { blas ->
        val empty = DenseVector.wrap(DoubleArray(0))

        assertEquals(0.0, blas.dotVector(empty, empty))
        assertEquals(0.0, blas.normVector(empty))
        assertEquals(0.0, blas.absoluteSumVector(empty))
        assertEquals(-1, blas.iamax(empty.values, empty.offset, empty.size, empty.stride))
    }

    @Test
    fun `symmetric rank two update agrees with the reference with both operands transposed`() = withVendor { blas ->
        val order = 3
        val depth = 2
        val a = matrix(depth, order, 34)
        val b = matrix(depth, order, 35)
        val c = DenseMatrix.wrap(order, order, vendorValues(order * order, 36))
        val expected = c.copyOf()
        ReferenceBlas.syr2k(1.25, a, b, true, 0.5, expected)

        blas.syr2k(1.25, a, b, true, 0.5, c, true)

        assertAgreesWithReference(expected.values, c.values, "syr2k transposed")
    }

    @Test
    fun `swap exchanges two interleaved vectors of one array`() = withVendor { blas ->
        // Two rows of a column-major matrix interleave in one array, which is what a pivot swap works on.
        val size = 4
        val lda = 3
        val storage = vendorValues(lda * size, 31)
        val expected = storage.copyOf()
        val first = StridedVector(storage, 0, size, lda)
        val second = StridedVector(storage, 1, size, lda)
        ScalarEngine.swap(expected, 0, expected, 1, size, lda, lda)

        blas.swap(first.values, first.offset, second.values, second.offset, first.size, first.stride, second.stride)

        assertContentEquals(expected, storage, "swap backing buffer")
    }

    @Test
    fun `the resolved library and version identify what actually ran`() = withVendor { blas ->
        assertTrue(blas.version.isNotEmpty(), "no version string")
        assertTrue(EngineImplementation.Gemm in blas.implementations, "gemm should be directly implemented")
    }

    @Test
    fun `the reported library is the file the symbol came from not the name that was asked for`() {
        for (vendor in Vendor.entries) {
            val blas = BuiltinEngines.vendor(vendor) ?: continue

            assertTrue(
                Path.of(blas.libraryPath).isAbsolute,
                "${vendor.vendorName} reported ${blas.libraryPath}, which is a candidate name rather than a file",
            )
            assertEquals(vendor, blas.vendor, "a library resolved under the wrong vendor")
        }
    }

    @Test
    fun `an opened library confirmed one compute thread where it can be asked`() {
        var checked = 0
        for (vendor in Vendor.entries) {
            val blas = BuiltinEngines.vendor(vendor) ?: continue
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
                        blas.axpyVector(alpha, x, y)
                    }
                    alpha to destination
                }
            }
            for (future in pool.invokeAll(work)) {
                val (alpha, actual) = future.get()
                val expected = DoubleArray(size)
                ScalarEngine.axpy(expected, 0, alpha, shared, 0, size)
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
                    repeat(REPEATS) { total += blas.dotVector(x, x) }
                    total / REPEATS
                }
            }
            val expected = ScalarEngine.dot(shared, 0, shared, 0, size)
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
        /** Calls per thread, as on Native: enough to interleave the threads, short enough for the test task. */
        const val REPEATS = 50
        const val TIMEOUT_SECONDS = 30L
    }
}
