package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import kotlin.test.*

class LinearAlgebraTest {

    @Test
    fun `gemv multiplies a matrix and its transpose by a vector`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0))) // 2x3
        assertTrue(doubleArrayOf(14.0, 32.0).contentEquals(koblas.gemv(a, doubleArrayOf(1.0, 2.0, 3.0))))
        assertTrue(
            doubleArrayOf(9.0, 12.0, 15.0).contentEquals(koblas.gemv(a, doubleArrayOf(1.0, 2.0), transpose = true)),
        )
    }

    @Test
    fun `gemm multiplies two matrices`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val b = DenseMatrix.of(arrayOf(doubleArrayOf(5.0, 6.0), doubleArrayOf(7.0, 8.0)))
        assertEquals(DenseMatrix.of(arrayOf(doubleArrayOf(19.0, 22.0), doubleArrayOf(43.0, 50.0))), a * b)
    }

    @Test
    fun `gemm with zero inner dimension yields a zero matrix`() {
        val a = DenseMatrix(2, 0)
        val b = DenseMatrix(0, 3)
        assertEquals(DenseMatrix(2, 3), a * b)
    }

    @Test
    fun `the dense routines reject mismatched shapes`() {
        assertFailsWith<IllegalArgumentException> { DenseMatrix(2, 3) * DenseMatrix(2, 2) }
        assertFailsWith<IllegalArgumentException> { koblas.gemv(DenseMatrix(2, 3), DoubleArray(2)) }
        assertFailsWith<IllegalArgumentException> {
            koblas.gemv(
                DenseMatrix(2, 3),
                DoubleArray(3),
                transpose = true,
            )
        }
    }
}
