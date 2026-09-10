package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinKernelProvider
import com.eignex.koblas.BuiltinKernels
import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.engine
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@OptIn(ExperimentalKoblasApi::class)
class SparseAlgorithmsTest {
    private val providers: List<BuiltinKernelProvider>
        get() = listOfNotNull(BuiltinKernels.scalar, BuiltinKernels.c, BuiltinKernels.simd).distinct()

    @Test
    fun `transposed gemv preserves csc reduction order`() {
        val values = DoubleArray(256)
        val matrix = singleColumn(values)
        val x = DoubleArray(256) { 1.0 }

        values[0] = 1e308
        values[1] = -1e308
        values[8] = 1e308
        for (provider in providers) {
            assertEquals(1e308, provider.engine().sparseBlas.gemv(matrix, x, transpose = true)[0])
        }

        values[1] = 1e308
        values[8] = -1e308
        for (provider in providers) {
            assertEquals(Double.POSITIVE_INFINITY, provider.engine().sparseBlas.gemv(matrix, x, transpose = true)[0])
        }
    }

    @Test
    fun `gemv updates do not fuse multiplication and addition`() {
        val matrix = singleColumn(DoubleArray(256) { 1e308 })
        for (provider in providers) {
            val y = DoubleArray(256) { -1e308 }
            provider.engine().sparseBlas.gemv(1.0, matrix, doubleArrayOf(2.0), 1.0, y, transpose = false)
            assertTrue(y.all { it == Double.POSITIVE_INFINITY }, provider.engine().name)
        }
    }

    @Test
    fun `gemv evaluates zero coefficient updates`() {
        val matrix = singleColumn(doubleArrayOf(Double.POSITIVE_INFINITY))
        for (provider in providers) {
            val engine = provider.engine()
            assertTrue(engine.sparseBlas.gemv(matrix, doubleArrayOf(0.0))[0].isNaN(), engine.name)

            val y = doubleArrayOf(0.0)
            engine.sparseBlas.gemv(Double.MIN_VALUE, matrix, doubleArrayOf(0.5), 0.0, y, transpose = false)
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
