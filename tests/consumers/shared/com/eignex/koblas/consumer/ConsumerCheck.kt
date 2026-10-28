package com.eignex.koblas.consumer

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import com.eignex.koblas.vendor.openBlas
import kotlin.math.abs

/** Checks the published dependency through its public API on each consuming runtime. */
object ConsumerCheck {
    /** Exercises transpose products, workspace reuse, and writes through aliased caller storage. */
    fun run(expectSimd: Boolean = false) {
        check(koblas.vectorKernels.name.startsWith("simd(") == expectSimd)
        check(koblas.vendor == null)
        if (!expectSimd) check(openBlas() == null)
        val portable = KoblasEngine.portable()
        check(portable.vendor == null)
        check(portable.vectorKernels.name == "scalar")
        for (engine in listOf(portable, koblas)) {
            val workspace = Workspace()
            for ((rows, cols) in listOf(12 to 24, 24 to 48, 48 to 48)) {
                val a = DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { (it % 19 - 9) / 17.0 })
                val x = DoubleArray(cols) { (it % 7 - 3) / 9.0 }
                val y = DoubleArray(rows)
                val expectedY = referenceGemv(a, x)
                val transposed = DoubleArray(cols)
                val expectedTransposed = referenceGemv(a, expectedY, transpose = true)
                val gram = DenseMatrix.zero(cols, cols)
                val expectedGram = referenceProduct(a, a, transposeA = true)
                var firstY: DoubleArray? = null
                var firstTransposed: DoubleArray? = null
                var firstGram: DoubleArray? = null
                repeat(8) {
                    engine.gemv(1.0, a, x, 0.0, y, workspace = workspace)
                    engine.gemv(1.0, a, y, 0.0, transposed, transpose = true, workspace = workspace)
                    engine.gemm(1.0, a, true, a, false, 0.0, gram, workspace)
                    assertAgreesWithReference(expectedY, y)
                    assertAgreesWithReference(expectedTransposed, transposed)
                    assertAgreesWithReference(expectedGram.values, gram.values)
                    if (engine === portable) {
                        firstY?.let { assertSameBits(it, y) }
                        firstTransposed?.let { assertSameBits(it, transposed) }
                        firstGram?.let { assertSameBits(it, gram.values) }
                        if (firstY == null) {
                            firstY = y.copyOf()
                            firstTransposed = transposed.copyOf()
                            firstGram = gram.values.copyOf()
                        }
                    }
                }
            }
            val matrix = DenseMatrix.wrap(2, 2, doubleArrayOf(2.0, 3.0, 5.0, 7.0))
            val storage = doubleArrayOf(2.0, 3.0)
            val expected = referenceGemv(matrix, storage.copyOf())
            engine.gemv(1.0, matrix, storage, 0.0, storage, workspace = workspace)
            assertAgreesWithReference(expected, storage)
        }
        if (expectSimd) {
            openBlas()?.let { host ->
                val actual = DenseMatrix.zero(2, 2)
                val matrix = DenseMatrix.wrap(2, 2, doubleArrayOf(2.0, 3.0, 5.0, 7.0))
                val expectedProduct = referenceProduct(matrix, matrix)
                host.gemm(1.0, matrix, false, matrix, false, 0.0, actual)
                assertAgreesWithReference(expectedProduct.values, actual.values)
                println("Published FFM binding: ${host.vendor}")
            }
        }
    }

    private fun referenceGemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val rows = if (transpose) a.cols else a.rows
        val depth = if (transpose) a.rows else a.cols
        return DoubleArray(rows) { row ->
            var sum = 0.0
            for (k in 0 until depth) sum += (if (transpose) a[k, row] else a[row, k]) * x[k]
            sum
        }
    }

    private fun referenceProduct(a: DenseMatrix, b: DenseMatrix, transposeA: Boolean = false): DenseMatrix {
        val rows = if (transposeA) a.cols else a.rows
        val depth = if (transposeA) a.rows else a.cols
        check(depth == b.rows)
        return DenseMatrix.wrap(rows, b.cols, DoubleArray(rows * b.cols) { index ->
            val row = index % rows
            val column = index / rows
            var sum = 0.0
            for (k in 0 until depth) sum += (if (transposeA) a[k, row] else a[row, k]) * b[k, column]
            sum
        })
    }

    private fun assertSameBits(expected: DoubleArray, actual: DoubleArray) {
        check(expected.size == actual.size)
        for (i in expected.indices) check(expected[i].toRawBits() == actual[i].toRawBits()) {
            "Portable result changed at entry $i"
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
