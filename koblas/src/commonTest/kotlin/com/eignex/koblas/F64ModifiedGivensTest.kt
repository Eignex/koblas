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
                -1.0, 5.9604644775390625e-8, -2.44140625e-4, 1.0,
                5.9604644775390625e-8,
            ),
            Case(
                1e20, 1e20, 2.0, 1.0,
                284217.09430404007, 284217.09430404007, 41943040.0,
                -1.0, 16777216.0, -4096.0, 1.0, 16777216.0,
            ),
        )

        for (case in cases) {
            assertModifiedGivensEquals(case, rotmg(case.inputD1, case.inputD2, case.inputX1, case.y1))
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
