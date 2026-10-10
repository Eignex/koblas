package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.assertClose
import com.eignex.koblas.dense.simdAvailable
import org.junit.Assume
import kotlin.random.Random
import kotlin.test.Test

class SparseSimdTest {
    @Test
    fun `indexed scatter agrees with the portable implementation`() {
        forEachPattern(::assertScatterAgreesWithReference)
    }

    @Test
    fun `indexed gather agrees with the portable implementation`() {
        forEachPattern(::assertGatherAgreesWithReference)
    }

    @Test
    fun `indexed gather zero agrees with the portable implementation`() {
        forEachPattern(::assertGatherZeroAgreesWithReference)
    }

    @Test
    fun `indexed axpy agrees with the portable implementation`() {
        forEachPattern { x, dense -> assertAxpyAgreesWithReference(x, dense, -0.75) }
    }

    @Test
    fun `indexed axpy rounds multiplication before addition`() {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        val count = 256
        val x = SparseVector.wrap(count, IntArray(count) { it }, DoubleArray(count) { 1e308 })
        val destination = DoubleArray(count) { -1e308 }

        assertAxpyAgreesWithReference(x, destination, 2.0)
    }

    private fun assertScatterAgreesWithReference(x: SparseVector, dense: DoubleArray) {
        val expected = dense.copyOf()
        ReferenceSparseBlas.scatter(x, expected)
        val actual = dense.copyOf()

        SparseSimd.scatter(x.indices, 0, x.values, 0, x.values.size, actual)

        assertClose(expected, actual, "scatter nnz=${x.values.size}")
    }

    private fun assertGatherAgreesWithReference(x: SparseVector, dense: DoubleArray) {
        val expected = SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
        ReferenceSparseBlas.gather(expected, dense.copyOf())
        val actual = SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())

        SparseSimd.gather(actual.indices, 0, actual.values, 0, actual.values.size, dense.copyOf())

        assertClose(expected.values, actual.values, "gather nnz=${x.values.size}")
    }

    private fun assertGatherZeroAgreesWithReference(x: SparseVector, dense: DoubleArray) {
        val expectedX = SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
        val expectedDense = dense.copyOf()
        ReferenceSparseBlas.gatherZero(expectedX, expectedDense)
        val actualX = SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
        val actualDense = dense.copyOf()

        SparseSimd.gatherZero(actualX.indices, 0, actualX.values, 0, actualX.values.size, actualDense)

        assertClose(expectedX.values, actualX.values, "gatherZero values nnz=${x.values.size}")
        assertClose(expectedDense, actualDense, "gatherZero dense nnz=${x.values.size}")
    }

    private fun assertAxpyAgreesWithReference(x: SparseVector, dense: DoubleArray, alpha: Double) {
        val expected = dense.copyOf()
        ReferenceSparseBlas.axpy(expected, alpha, x)
        val actual = dense.copyOf()

        SparseSimd.axpy(x.indices, 0, x.values, 0, x.values.size, alpha, actual)

        assertClose(expected, actual, "axpy nnz=${x.values.size}")
    }

    private fun forEachPattern(block: (SparseVector, DoubleArray) -> Unit) {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        for (nnz in intArrayOf(1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 255, 256, 257)) {
            val random = Random(nnz)
            val size = 2 * nnz + 1
            val x = SparseVector.wrap(
                size,
                IntArray(nnz) { 2 * it + 1 },
                DoubleArray(nnz) { random.nextDouble(-1.0, 1.0) },
            )
            block(x, DoubleArray(size) { random.nextDouble(-1.0, 1.0) })
        }
    }
}
