package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ScalarBlockKernelsTest {
    @Test
    fun `direct and packed modes agree with independent strided scalar arithmetic`() {
        for ((m, n, k) in listOf(Triple(1, 3, 1), Triple(3, 2, 5), Triple(5, 4, 3))) {
            val a = MatrixWindow(DoubleArray(m * k) { (it % 7 - 3) * 0.25 }, m, k)
            val b = MatrixWindow(DoubleArray(k * n) { (it % 5 - 2) * 0.5 }, k, n)
            for (leftPacked in listOf(false, true)) {
                for (rightPacked in listOf(false, true)) {
                    val left = if (leftPacked) {
                        ScalarLayoutKernels.pack(
                            a,
                            PackedMatrixLayout(PackedRole.Left, m, k, 4),
                        )
                    } else {
                        a
                    }
                    val right = if (rightPacked) {
                        ScalarLayoutKernels.pack(
                            b,
                            PackedMatrixLayout(PackedRole.Right, k, n, 3),
                        )
                    } else {
                        b
                    }
                    for (alpha in listOf(0.0, -0.5, 1.0, 2.0)) {
                        for (beta in listOf(0.0, 1.0, -0.5)) {
                            assertProductAgreesWithReference(a, b, left, right, alpha, beta)
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `retained operands reuse unscaled values for different alpha`() {
        val a = MatrixWindow(doubleArrayOf(1.0, 2.0, 3.0, 4.0), 2, 2)
        val b = MatrixWindow(doubleArrayOf(2.0, 3.0, 4.0, 5.0), 2, 2)
        val packed = ScalarLayoutKernels.pack(b, PackedMatrixLayout(PackedRole.Right, 2, 2, 3))
        for (alpha in listOf(1.0, -2.0, 0.25)) assertProductAgreesWithReference(a, b, a, packed, alpha, 0.0)
    }

    @Test
    fun `zero alpha and zero depth scale only selected output`() {
        val untouched = Double.fromBits(0x7ff8000000000042L)
        for (depth in listOf(0, 2)) {
            val a = MatrixWindow(DoubleArray(2 * depth) { Double.NaN }, 2, depth)
            val b = MatrixWindow(DoubleArray(depth * 2) { Double.NaN }, depth, 2)
            val data = DoubleArray(8) { untouched }
            ScalarBlockKernels.product(
                a,
                b,
                MatrixWindow(data, 2, 2, 1, columnStride = 3),
                alpha = 0.0,
                output = BlockOutput.lower(),
            )
            for (index in data.indices) {
                assertEquals(if (index in listOf(1, 2, 5)) 0L else untouched.toRawBits(), data[index].toRawBits())
            }
        }
    }

    @Test
    fun `beta zero does not propagate original output and depth continuation applies beta once`() {
        val a = MatrixWindow(doubleArrayOf(2.0), 1, 1)
        val b = MatrixWindow(doubleArrayOf(3.0), 1, 1)
        val data = doubleArrayOf(Double.NaN)
        val c = MatrixWindow(data, 1, 1)
        ScalarBlockKernels.product(a, b, c, beta = 0.0)
        ScalarBlockKernels.product(a, b, c, beta = Double.NaN, contribution = DepthContribution.Subsequent)
        assertEquals(12.0, data[0])
    }

    @Test
    fun `alias staging protects both input sides and selected output`() {
        for (aliasLeft in listOf(false, true)) {
            val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 99.0)
            val other = doubleArrayOf(5.0, 6.0, 7.0, 8.0)
            val a = MatrixWindow(if (aliasLeft) data else other, 2, 2)
            val b = MatrixWindow(if (aliasLeft) other else data, 2, 2)
            val expected = data.copyOf()
            val c = MatrixWindow(data, 2, 2)
            val separate = MatrixWindow(expected, 2, 2)
            ScalarBlockKernels.product(
                snapshotOperand(a),
                snapshotOperand(b),
                separate,
                beta = 0.5,
                output = BlockOutput.upper(),
            )

            val scratch = DoubleArray(ScalarBlockKernels.scratchSize(a, b, c))
            ScalarBlockKernels.product(a, b, c, beta = 0.5, output = BlockOutput.upper(), scratch = scratch)

            assertContentEquals(expected, data)
        }
    }

    @Test
    fun `context role and scratch failures precede output mutation`() {
        val layout = PackedMatrixLayout(PackedRole.Left, 2, 2, 2, requiredSvlBytes = 64)
        val packed = ScalarLayoutKernels.pack(MatrixWindow(DoubleArray(4) { 1.0 }, 2, 2), layout)
        val data = DoubleArray(4) { -0.0 }
        val c = MatrixWindow(data, 2, 2)
        val b = MatrixWindow(DoubleArray(4) { 1.0 }, 2, 2)
        assertFailsWith<IllegalArgumentException> { ScalarBlockKernels.product(packed, b, c, svlBytes = 32) }
        assertFailsWith<IllegalArgumentException> { ScalarBlockKernels.product(b, packed, c, svlBytes = 64) }
        assertFailsWith<IllegalArgumentException> { ScalarBlockKernels.product(c, b, c, scratch = DoubleArray(3)) }
        assertFailsWith<IllegalArgumentException> { ScalarBlockKernels.product(c, b, c, scratch = data) }
        assertContentEquals(LongArray(4) { Long.MIN_VALUE }, data.map { it.toRawBits() }.toLongArray())
    }

    @Test
    fun `ordered fallback preserves scaling overflow and raw zero skips`() {
        val a = MatrixWindow(doubleArrayOf(1e308), 1, 1)
        val b = MatrixWindow(doubleArrayOf(2.0), 1, 1)
        val data = DoubleArray(1)
        ScalarBlockKernels.product(
            a,
            b,
            MatrixWindow(data, 1, 1),
            alpha = 0.5,
            evaluation = ProductEvaluation.OrderedUpdates,
        )
        assertEquals(1e308, data[0])
        ScalarBlockKernels.product(a, b, MatrixWindow(data, 1, 1), alpha = 0.5)
        assertEquals(Double.POSITIVE_INFINITY, data[0])
        val nan = MatrixWindow(doubleArrayOf(Double.NaN), 1, 1)
        val zero = MatrixWindow(doubleArrayOf(0.0), 1, 1)
        ScalarBlockKernels.product(nan, zero, MatrixWindow(data, 1, 1), evaluation = ProductEvaluation.OrderedUpdates)
        assertEquals(0.0, data[0])
        ScalarBlockKernels.product(
            nan,
            zero,
            MatrixWindow(data, 1, 1),
            evaluation = ProductEvaluation.ArithmeticUpdates,
        )
        assertEquals(Double.NaN, data[0])
    }

    @Test
    fun `output diagonal offset masks a block in global coordinates`() {
        val data = DoubleArray(6) { -1.0 }
        ScalarBlockKernels.product(
            MatrixWindow(DoubleArray(2) { 1.0 }, 2, 1),
            MatrixWindow(DoubleArray(3) { 1.0 }, 1, 3),
            MatrixWindow(data, 2, 3),
            output = BlockOutput.lower(-1),
        )
        assertContentEquals(doubleArrayOf(-1.0, 1.0, -1.0, -1.0, -1.0, -1.0), data)
    }
}

private fun assertProductAgreesWithReference(
    a: MatrixWindow,
    b: MatrixWindow,
    left: MatrixOperand,
    right: MatrixOperand,
    alpha: Double,
    beta: Double,
) {
    val m = a.rows
    val n = b.columns
    val actual = DoubleArray((m + 2) * n + 2) { (it - 3) * 0.25 }
    val expected = actual.copyOf()
    val c = MatrixWindow(actual, m, n, 1, columnStride = m + 2)
    for (j in 0 until n) {
        for (i in 0 until m) {
            val index = 1 + i + j * (m + 2)
            val original = if (beta == 0.0) 0.0 else beta * expected[index]
            if (alpha == 0.0) {
                expected[index] = original
            } else {
                var dot = 0.0
                for (p in 0 until a.columns) dot += a.data[i + p * m] * b.data[p + j * b.rows]
                expected[index] = alpha * dot + original
            }
        }
    }
    ScalarBlockKernels.product(left, right, c, alpha, beta)
    assertClose(expected, actual, "packed block versus independent scalar")
}
