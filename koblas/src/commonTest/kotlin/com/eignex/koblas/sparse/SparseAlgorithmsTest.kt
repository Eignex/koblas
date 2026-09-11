package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SparseAlgorithmsTest {
    private val engines: List<KoblasEngine>
        get() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.c, BuiltinEngines.simd).distinct()

    @Test
    fun `gemv agrees across transpose alpha and beta variants`() {
        val matrix = SparseMatrix.ofTriplets(
            rows = 5,
            cols = 5,
            rowIdx = intArrayOf(0, 2, 1, 3, 4, 2, 0),
            colIdx = intArrayOf(0, 0, 1, 1, 2, 3, 4),
            values = doubleArrayOf(2.0, -1.0, 3.0, 0.5, 4.0, -2.0, 1.5),
        )
        val rng = Random(20260822)
        for (transpose in booleanArrayOf(false, true)) {
            for (alpha in doubleArrayOf(0.0, 1.0, -0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                    val x = DoubleArray(5) { rng.nextDouble(-1.0, 1.0) }
                    val initial = DoubleArray(5) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = DoubleArray(5) { i ->
                        var product = 0.0
                        for (j in 0 until 5) product += (if (transpose) matrix[j, i] else matrix[i, j]) * x[j]
                        alpha * product + if (beta == 0.0) 0.0 else beta * initial[i]
                    }
                    for (engine in engines) {
                        val actual = initial.copyOf()
                        engine.gemv(alpha, matrix, x, beta, actual, transpose)
                        assertClose(expected, actual, "${engine.name} transpose=$transpose alpha=$alpha beta=$beta")
                    }
                }
            }
        }
    }

    @Test
    fun `transposed gemv preserves csc reduction order`() {
        val values = DoubleArray(256)
        val matrix = singleColumn(values)
        val x = DoubleArray(256) { 1.0 }

        values[0] = 1e308
        values[1] = -1e308
        values[8] = 1e308
        for (engine in engines) {
            assertEquals(1e308, engine.gemv(matrix, x, transpose = true)[0])
        }

        values[1] = 1e308
        values[8] = -1e308
        for (engine in engines) {
            assertEquals(Double.POSITIVE_INFINITY, engine.gemv(matrix, x, transpose = true)[0])
        }
    }

    @Test
    fun `gemv updates do not fuse multiplication and addition`() {
        val matrix = singleColumn(DoubleArray(256) { 1e308 })
        for (engine in engines) {
            val y = DoubleArray(256) { -1e308 }
            engine.gemv(1.0, matrix, doubleArrayOf(2.0), 1.0, y, transpose = false)
            assertTrue(y.all { it == Double.POSITIVE_INFINITY }, engine.name)
        }
    }

    @Test
    fun `gemv evaluates zero coefficient updates`() {
        val matrix = singleColumn(doubleArrayOf(Double.POSITIVE_INFINITY))
        for (engine in engines) {
            assertTrue(engine.gemv(matrix, doubleArrayOf(0.0))[0].isNaN(), engine.name)

            val y = doubleArrayOf(0.0)
            engine.gemv(Double.MIN_VALUE, matrix, doubleArrayOf(0.5), 0.0, y, transpose = false)
            assertTrue(y[0].isNaN(), engine.name)
        }
    }

    private fun singleColumn(values: DoubleArray): SparseMatrix = SparseMatrix.wrap(
        values.size,
        1,
        intArrayOf(0, values.size),
        IntArray(values.size) { it },
        values,
    )
}
