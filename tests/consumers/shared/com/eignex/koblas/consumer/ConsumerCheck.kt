package com.eignex.koblas.consumer

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngineApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.vendor.openBlas
import kotlin.math.abs

/** Checks the published dependency through its public API on each consuming runtime. */
@OptIn(KoblasEngineApi::class)
object ConsumerCheck {
    /** Exercises transpose products, workspace reuse, and writes through aliased caller storage. */
    fun run(expectSimd: Boolean = false) {
        check((BuiltinEngines.simd != null) == expectSimd)
        check(koblas.vendor == null)
        if (!expectSimd) check(openBlas() == null)
        val reference = BuiltinEngines.scalar
        val workspace = Workspace()
        for ((rows, cols) in listOf(12 to 24, 24 to 48, 48 to 48)) {
            val a = DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { (it % 19 - 9) / 16.0 })
            val x = DoubleArray(cols) { (it % 7 - 3) / 8.0 }
            val y = DoubleArray(rows)
            val expectedY = reference.gemv(a, x)
            val transposed = DoubleArray(cols)
            val expectedTransposed = reference.gemv(a, expectedY, transpose = true)
            val gram = DenseMatrix.zero(cols, cols)
            val expectedGram = DenseMatrix.zero(cols, cols)
            reference.gemm(1.0, a, true, a, false, 0.0, expectedGram)
            repeat(8) {
                koblas.gemv(1.0, a, x, 0.0, y, workspace = workspace)
                koblas.gemv(1.0, a, y, 0.0, transposed, transpose = true, workspace = workspace)
                koblas.gemm(1.0, a, true, a, false, 0.0, gram, workspace)
                assertAgreesWithReference(expectedY, y)
                assertAgreesWithReference(expectedTransposed, transposed)
                assertAgreesWithReference(expectedGram.values, gram.values)
            }
        }
        val matrix = DenseMatrix.wrap(2, 2, doubleArrayOf(2.0, 3.0, 5.0, 7.0))
        val storage = doubleArrayOf(2.0, 3.0)
        val expected = reference.gemv(matrix, storage.copyOf())
        koblas.gemv(1.0, matrix, storage, 0.0, storage, workspace = workspace)
        assertAgreesWithReference(expected, storage)
        if (expectSimd) {
            openBlas()?.let { host ->
                val actual = DenseMatrix.zero(2, 2)
                val expectedProduct = reference.gemm(matrix, matrix)
                host.gemm(1.0, matrix, false, matrix, false, 0.0, actual)
                assertAgreesWithReference(expectedProduct.values, actual.values)
                println("Published FFM binding: ${host.vendor}")
            }
        }
    }

    private fun assertAgreesWithReference(expected: DoubleArray, actual: DoubleArray) {
        check(expected.size == actual.size)
        for (i in expected.indices) {
            check(abs(expected[i] - actual[i]) <= 1e-11 * (1.0 + abs(expected[i]))) {
                "Entry $i: expected ${expected[i]}, got ${actual[i]}"
            }
        }
    }
}
