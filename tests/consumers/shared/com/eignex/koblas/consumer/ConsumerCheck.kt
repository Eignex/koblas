package com.eignex.koblas.consumer

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.Vendor
import kotlin.math.abs

/** Checks the published dependency through its public API on each consuming runtime. */
object ConsumerCheck {
    /** Exercises transpose products, workspace reuse, and writes through aliased caller storage. */
    fun run(expectSimd: Boolean = false) {
        check((BuiltinEngines.simd != null) == expectSimd)
        if (!expectSimd) {
            check(koblas.providers.none { it.vendor != null })
            check(Vendor.entries.firstNotNullOfOrNull { BuiltinEngines.vendor(it) } == null)
        }
        checkPartialProvider()
        val portable = BuiltinEngines.scalar
        check(portable.vendor == null)
        check(portable.name == "scalar")
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
                repeat(8) {
                    engine.gemv(1.0, a, x, 0.0, y, workspace = workspace)
                    engine.gemv(1.0, a, y, 0.0, transposed, transpose = true, workspace = workspace)
                    engine.gemm(1.0, a, true, a, false, 0.0, gram, workspace)
                    assertAgreesWithReference(expectedY, y)
                    assertAgreesWithReference(expectedTransposed, transposed)
                    assertAgreesWithReference(expectedGram.values, gram.values)
                }
            }
            val matrix = DenseMatrix.wrap(2, 2, doubleArrayOf(2.0, 3.0, 5.0, 7.0))
            val storage = doubleArrayOf(2.0, 3.0)
            val expected = referenceGemv(matrix, storage.copyOf())
            engine.gemv(1.0, matrix, storage, 0.0, storage, workspace = workspace)
            assertAgreesWithReference(expected, storage)
            val source = SparseMatrix.ofTriplets(2, 2, intArrayOf(0, 1), intArrayOf(0, 1), doubleArrayOf(2.0, 3.0))
            val sparse: Matrix = source
            val identity: Matrix = DenseMatrix.diagonal(2)
            val parent = DenseMatrix.wrap(6, 6, DoubleArray(36) { -99.0 })
            val destination = parent.view(2, 3, 2, 2)
            val expectedProduct = referenceProduct(DenseMatrix.ofRows(source.toArray()), identity as DenseMatrix)
            engine.gemm(1.0, sparse, false, identity, false, 0.0, destination, workspace)
            for (j in 0 until 6) for (i in 0 until 6) {
                val expectedEntry = if (i in 2..3 && j in 3..4) expectedProduct[i - 2, j - 3] else -99.0
                assertAgreesWithReference(doubleArrayOf(expectedEntry), doubleArrayOf(parent[i, j]))
            }
            val reversed = StridedVector(doubleArrayOf(3.0, 0.0, 2.0), 2, 2, -2)
            val output = DoubleArray(5) { -99.0 }
            val outputView = StridedVector(output, 3, 2, -2)
            engine.gemv(1.0, matrix, reversed, 0.0, outputView, workspace = workspace)
            val expectedStrided = referenceGemv(matrix, reversed.toDoubleArray())
            assertAgreesWithReference(doubleArrayOf(-99.0, expectedStrided[1], -99.0, expectedStrided[0], -99.0), output)
        }
        if (expectSimd) {
            Vendor.entries.firstNotNullOfOrNull { BuiltinEngines.vendor(it) }?.let { host ->
                val actual = DenseMatrix.zero(2, 2)
                val matrix = DenseMatrix.wrap(2, 2, doubleArrayOf(2.0, 3.0, 5.0, 7.0))
                val expectedProduct = referenceProduct(matrix, matrix)
                host.gemm(1.0, matrix, false, matrix, false, 0.0, actual)
                assertAgreesWithReference(expectedProduct.values, actual.values)
                println("Published FFM binding: ${host.vendor}")
            }
        }
    }

    private fun checkPartialProvider() {
        var called = false
        val provider = object : KoblasEngine {
            override val name = "consumer gemm"
            override val implementations = setOf(EngineImplementation.Gemm)
            override fun computeGemm(
                alpha: Double, a: DenseMatrixRegion, transposeA: Boolean, b: DenseMatrixRegion,
                transposeB: Boolean, beta: Double, c: DenseMatrixRegion,
            ) {
                called = true
                BuiltinEngines.scalar.gemm(alpha, a, transposeA, b, transposeB, beta, c)
            }
        }
        val engine = KoblasEngine.combine(
            EngineProfile("consumer", listOf(EngineRule(provider.id, provider.implementations))), provider,
        )
        val a = DenseMatrix.wrap(2, 2, doubleArrayOf(0.5, -2.0, 3.0, 1.0))
        val result = DenseMatrix.zero(2, 2)
        engine.gemm(1.0, a, false, a, false, 0.0, result)
        check(called)
        assertAgreesWithReference(referenceProduct(a, a).values, result.values)
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

    private fun assertAgreesWithReference(expected: DoubleArray, actual: DoubleArray) {
        check(expected.size == actual.size)
        for (i in expected.indices) {
            check(abs(expected[i] - actual[i]) <= 1e-11 * (1.0 + abs(expected[i]))) {
                "Entry $i: expected ${expected[i]}, got ${actual[i]}"
            }
        }
    }
}
