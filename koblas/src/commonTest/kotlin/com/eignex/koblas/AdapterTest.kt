package com.eignex.koblas

import com.eignex.koblas.*
import kotlin.test.Test
import kotlin.test.assertEquals

class AdapterTest {
    @Test
    fun `gemv accepts a foreign matrix and vector`() {
        val a = ForeignSpdMatrix(5)
        val x = ForeignRampVector(5)
        val viaAdapter = a * x
        val viaStorage = DenseMatrix.ofRows(a.toArray()) * DenseVector.of(x.toDoubleArray())
        for (i in 0 until 5) {
            assertEquals(viaStorage[i], viaAdapter[i], 1e-12, "row $i")
        }
    }

    @Test
    fun `the vector reductions accept a foreign vector`() {
        val x = ForeignRampVector(9)
        val dense = DenseVector.of(x.toDoubleArray())
        assertEquals(dense.norm2(), x.norm2(), 1e-12)
        assertEquals(dense.asum(), x.asum(), 1e-12)
        assertEquals(dense.iamax(), x.iamax())
        assertEquals(dense dot dense, x dot x, 1e-12)
        assertEquals(dense dot dense, dense dot x, 1e-12)
        assertEquals(dense dot dense, x dot dense, 1e-12)
    }

    @Test
    fun `axpy and copy accept a foreign source`() {
        val x = ForeignRampVector(6)
        val y = DenseVector(6)
        y.axpy(2.0, x)
        for (i in 0 until 6) assertEquals(2.0 * x[i], y.data[i], 1e-12, "axpy at $i")

        val dst = DenseVector(6)
        copy(x, dst)
        for (i in 0 until 6) assertEquals(x[i], dst.data[i], 1e-12, "copy at $i")
    }

    @Test
    fun `a sparse operand still walks its stored entries against an adapter`() {
        val sparse = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))
        val ramp = ForeignRampVector(6)
        var expected = 0.0
        for (i in 0 until 6) expected += sparse[i] * ramp[i]
        assertEquals(expected, sparse dot ramp, 1e-12)
        assertEquals(expected, ramp dot sparse, 1e-12)
    }
}
