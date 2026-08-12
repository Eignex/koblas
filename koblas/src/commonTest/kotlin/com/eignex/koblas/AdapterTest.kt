package com.eignex.koblas

import com.eignex.koblas.dense.cholesky
import com.eignex.koblas.dense.solve
import kotlin.test.Test
import kotlin.test.assertEquals

class AdapterTest {

    private class Spd(override val rows: Int) : MatrixLike {
        override val cols: Int get() = rows
        override fun get(i: Int, j: Int): Double = if (i == j) rows + 2.0 else 1.0 / (1 + i + j)
        override fun toArray(): Array<DoubleArray> = Array(rows) { i -> DoubleArray(cols) { j -> this[i, j] } }
    }

    private class Ramp(override val size: Int) : VectorLike {
        override fun get(i: Int): Double = i * 0.5 - 1.0
        override fun toDoubleArray(): DoubleArray = DoubleArray(size) { this[it] }
    }

    @Test
    fun `gemv accepts a foreign matrix and vector`() {
        val a = Spd(5)
        val x = Ramp(5)
        val viaAdapter = a.matVec(x)
        val viaStorage = DenseMatrix.of(a.toArray()).matVec(DenseVector.of(x.toDoubleArray()))
        for (i in 0 until 5) {
            assertEquals(viaStorage[i], viaAdapter[i], 1e-12, "row $i")
        }
    }

    @Test
    fun `cholesky and the SPD solve accept a foreign matrix once it is materialised`() {
        val a = Spd(4)
        val l = DenseMatrix.of(a.toArray()).cholesky()
        val b = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val x = l.solve(b)
        for (i in 0 until 4) {
            var s = 0.0
            for (j in 0 until 4) s += a[i, j] * x[j]
            assertEquals(b[i], s, 1e-9, "residual at $i")
        }
    }

    @Test
    fun `the vector reductions accept a foreign vector`() {
        val x = Ramp(9)
        val dense = DenseVector.of(x.toDoubleArray())
        assertEquals(norm2(dense), norm2(x), 1e-12)
        assertEquals(asum(dense), asum(x), 1e-12)
        assertEquals(iamax(dense), iamax(x))
        assertEquals(dense dot dense, x dot x, 1e-12)
        assertEquals(dense dot dense, dense dot x, 1e-12)
        assertEquals(dense dot dense, x dot dense, 1e-12)
    }

    @Test
    fun `axpy and copy accept a foreign source`() {
        val x = Ramp(6)
        val y = DenseVector(6)
        axpy(y, 2.0, x)
        for (i in 0 until 6) assertEquals(2.0 * x[i], y.data[i], 1e-12, "axpy at $i")

        val dst = DenseVector(6)
        copy(x, dst)
        for (i in 0 until 6) assertEquals(x[i], dst.data[i], 1e-12, "copy at $i")
    }

    @Test
    fun `a sparse operand still walks its stored entries against an adapter`() {
        val sparse = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))
        val ramp = Ramp(6)
        var expected = 0.0
        for (i in 0 until 6) expected += sparse[i] * ramp[i]
        assertEquals(expected, sparse dot ramp, 1e-12)
        assertEquals(expected, ramp dot sparse, 1e-12)
    }
}
