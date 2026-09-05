package com.eignex.koblas

import com.eignex.koblas.core.F64DenseVector
import com.eignex.koblas.core.F64StridedVectorView
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class F64ModifiedGivensTest {

    @Test
    fun `rotmg matches Netlib branch states`() {
        val cases = listOf(
            Case(-1.0, 3.0, 2.0, 4.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
            Case(1.0, 0.0, 2.0, 4.0, 1.0, 0.0, 2.0, -2.0, 1.0, 0.0, 0.0, 1.0),
            Case(1.0, 1.0, 2.0, 1.0, 0.8, 0.8, 2.5, 0.0, 1.0, -0.5, 0.5, 1.0),
            Case(1.0, 1.0, 1.0, 2.0, 0.8, 0.8, 2.5, 1.0, 0.5, -1.0, 1.0, 0.5),
            Case(1.0, -1.0, 1.0, 2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
        )

        for (case in cases) {
            val actual = rotmg(case.inputD1, case.inputD2, case.inputX1, case.y1)
            assertModifiedGivensEquals(case, actual)
        }
    }

    @Test
    fun `rotmg matches Netlib rescaling states`() {
        val cases = listOf(
            Case(
                1e-20, 1e-20, 2.0, 1.0,
                2.2517998136852478e-6, 2.2517998136852478e-6, 1.4901161193847656e-7,
                -1.0, 5.960464477539063e-8, -2.9802322387695312e-8, 2.9802322387695312e-8,
                5.960464477539063e-8,
            ),
            Case(
                1e20, 1e20, 2.0, 1.0,
                284217.09430404007, 284217.09430404007, 41943040.0,
                -1.0, 16777216.0, -8388608.0, 8388608.0, 16777216.0,
            ),
        )

        for (case in cases) {
            assertModifiedGivensEquals(case, rotmg(case.inputD1, case.inputD2, case.inputX1, case.y1))
        }
    }

    @Test
    fun `rotmg eliminates the second component across every rescaling path`() {
        // The defining property, stated without hand-written constants so it holds however the
        // rescaling loops are reorganized: H maps the unscaled pair to the returned x1 and a zero.
        val inputs = listOf(
            doubleArrayOf(1.0, 1.0, 2.0, 1.0),
            doubleArrayOf(1.0, 1.0, 1.0, 2.0),
            doubleArrayOf(1e-9, 1e-9, 1.0, 2.0),
            doubleArrayOf(1e-20, 1e-20, 2.0, 1.0),
            doubleArrayOf(1e20, 1e20, 2.0, 1.0),
            doubleArrayOf(1e20, 1e-20, 2.0, 1.0),
            doubleArrayOf(1e-20, 1e20, 2.0, 1.0),
        )

        for (input in inputs) {
            val (d1, d2, x1, y1) = input
            val h = rotmg(d1, d2, x1, y1)
            val label = "d1=$d1 d2=$d2 x1=$x1 y1=$y1"

            val eliminated = h.h21 * x1 + h.h22 * y1
            val eliminatedScale = maxOf(kotlin.math.abs(h.h21 * x1), kotlin.math.abs(h.h22 * y1))
            assertEquals(0.0, eliminated, eliminatedScale * 1e-13, "second component for $label")

            val carried = h.h11 * x1 + h.h12 * y1
            assertEquals(h.x1, carried, tolerance(h.x1), "first component for $label")
        }
    }

    @Test
    fun `rotm applies modified Givens forms`() {
        val cases = listOf(
            rotmg(1.0, 0.0, 2.0, 1.0) to doubleArrayOf(2.0, -1.0, 1.0, 4.0),
            rotmg(1.0, 1.0, 2.0, 1.0) to doubleArrayOf(2.0, -1.0, 1.0, 4.0),
            rotmg(1.0, 1.0, 1.0, 2.0) to doubleArrayOf(2.0, -1.0, 1.0, 4.0),
            rotmg(-1.0, 1.0, 1.0, 2.0) to doubleArrayOf(2.0, -1.0, 1.0, 4.0),
        )

        for ((transformation, values) in cases) {
            val x = F64DenseVector.of(values.copyOfRange(0, 2))
            val y = F64DenseVector.of(values.copyOfRange(2, 4))
            val expectedX = DoubleArray(x.size) { transformation.h11 * x[it] + transformation.h12 * y[it] }
            val expectedY = DoubleArray(y.size) { transformation.h21 * x[it] + transformation.h22 * y[it] }

            rotm(x, y, transformation)

            assertContentEquals(expectedX, x.data)
            assertContentEquals(expectedY, y.data)
        }
    }

    @Test
    fun `rotm on aliased dense vectors keeps the final write from y`() {
        val buffer = doubleArrayOf(2.0, -1.0)
        val x = F64DenseVector.wrap(buffer)
        val y = F64DenseVector.wrap(buffer)
        val transformation = rotmg(1.0, 1.0, 2.0, 1.0)
        val expected = DoubleArray(buffer.size) {
            transformation.h21 * buffer[it] + transformation.h22 * buffer[it]
        }

        rotm(x, y, transformation)

        assertContentEquals(expected, buffer)
        assertContentEquals(expected, x.data)
        assertContentEquals(expected, y.data)
    }

    @Test
    fun `rotmg terminates for a non finite d1`() {
        // The point of this test is that it returns at all: the GAM rescaling loop cannot bring an
        // infinite d1 back into range, so it must stop on non-finiteness rather than spin forever.
        val transformation = rotmg(Double.POSITIVE_INFINITY, 1.0, 1.0, 1.0)

        assertEquals(listOf(-2.0, -1.0, 0.0, 1.0).contains(transformation.flag), true)
    }

    @Test
    fun `rotm snapshots overlapping strided inputs`() {
        val buffer = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val x = F64StridedVectorView(buffer, offset = 0, size = 3)
        val y = F64StridedVectorView(buffer, offset = 1, size = 3)
        val transformation = rotmg(1.0, 1.0, 2.0, 1.0)

        rotm(x, y, transformation)

        assertContentEquals(doubleArrayOf(2.0, 1.5, 2.0, 2.5, 5.0), buffer)
    }

    @Test
    fun `rotm supports negative strided views`() {
        val xData = doubleArrayOf(1.0, -1.0, 2.0, -2.0, 3.0)
        val yData = doubleArrayOf(4.0, -4.0, 5.0, -5.0, 6.0)
        val x = F64StridedVectorView(xData, offset = 4, size = 3, stride = -2)
        val y = F64StridedVectorView(yData, offset = 4, size = 3, stride = -2)
        val transformation = rotmg(1.0, 1.0, 1.0, 2.0)

        rotm(x, y, transformation)

        assertContentEquals(doubleArrayOf(4.5, -1.0, 6.0, -2.0, 7.5), xData)
        assertContentEquals(doubleArrayOf(1.0, -4.0, 0.5, -5.0, 0.0), yData)
    }

    private fun assertModifiedGivensEquals(expected: Case, actual: F64ModifiedGivens) {
        assertEquals(expected.d1, actual.d1, tolerance(expected.d1), "d1")
        assertEquals(expected.d2, actual.d2, tolerance(expected.d2), "d2")
        assertEquals(expected.x1, actual.x1, tolerance(expected.x1), "x1")
        assertEquals(expected.flag, actual.flag, "flag")
        assertEquals(expected.h11, actual.h11, tolerance(expected.h11), "h11")
        assertEquals(expected.h21, actual.h21, tolerance(expected.h21), "h21")
        assertEquals(expected.h12, actual.h12, tolerance(expected.h12), "h12")
        assertEquals(expected.h22, actual.h22, tolerance(expected.h22), "h22")
    }

    private fun tolerance(value: Double): Double = maxOf(1e-15, kotlin.math.abs(value) * 1e-14)

    private data class Case(
        val inputD1: Double,
        val inputD2: Double,
        val inputX1: Double,
        val y1: Double,
        val d1: Double,
        val d2: Double,
        val x1: Double,
        val flag: Double,
        val h11: Double,
        val h21: Double,
        val h12: Double,
        val h22: Double,
    )
}
