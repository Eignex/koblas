package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinKernels
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.SparseVector
import com.eignex.koblas.engine
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/** Correctness gate for the real benchmark-only oneMKL sparse adapter. */
@OptIn(ExperimentalKoblasApi::class)
class OneMklSparseComparatorTest {
    private val scalar = BuiltinKernels.scalar.engine()

    @Test
    fun `indexed level one and caller owned workspace agree with scalar oracle`() {
        val external = requiredOneMkl() ?: return
        val sparse = SparseVector.of(11, intArrayOf(1, 4, 8), doubleArrayOf(2.5, -3.0, 0.75))
        val dense = DoubleArray(11) { it * 0.25 - 1.0 }
        assertEquals(scalar.sparseKernels.dot(sparse, dense), external.dot(sparse, dense), 1e-13)

        val expectedAxpy = dense.copyOf()
        val actualAxpy = dense.copyOf()
        scalar.sparseKernels.axpy(expectedAxpy, -0.6, sparse)
        external.axpy(-0.6, sparse, actualAxpy)
        assertContentEquals(expectedAxpy, actualAxpy)

        val expectedScatter = DoubleArray(11)
        val actualScatter = DoubleArray(11)
        scalar.sparseKernels.scatter(sparse, expectedScatter)
        external.scatter(sparse, actualScatter)
        assertContentEquals(expectedScatter, actualScatter)

        val expectedGatherVector = SparseVector.of(11, sparse.copyIndices(), DoubleArray(3))
        val actualGather = DoubleArray(3)
        scalar.sparseKernels.gather(expectedGatherVector, dense)
        external.gather(sparse, dense, actualGather)
        assertContentEquals(expectedGatherVector.values, actualGather)

        val expectedSource = dense.copyOf()
        val actualSource = dense.copyOf()
        val expectedGatherZeroVector = SparseVector.of(11, sparse.copyIndices(), DoubleArray(3))
        scalar.sparseKernels.gatherZero(expectedGatherZeroVector, expectedSource)
        val actualGatherZero = DoubleArray(3)
        external.gatherZero(sparse, actualSource, actualGatherZero)
        assertContentEquals(expectedGatherZeroVector.values, actualGatherZero)
        assertContentEquals(expectedSource, actualSource)

        val indices = intArrayOf(91, 7, 2, 9, 4, 82)
        val values = doubleArrayOf(91.0, 1.5, -2.0, 3.25, -4.5, 82.0)
        val expectedAccumulator = DoubleArray(12) { 123.0 }.also { it[2] = 8.0 }
        val actualAccumulator = expectedAccumulator.copyOf()
        val expectedMarks = IntArray(12).also { it[2] = 6 }
        val actualMarks = expectedMarks.copyOf()
        val expectedTouched = IntArray(10) { -1 }.also { it[2] = 2 }
        val actualTouched = expectedTouched.copyOf()
        val expectedCount = com.eignex.koblas.sparse.SparseWorkspace.scatterAxpy(
            0.75, indices, 1, values, 1, 4,
            expectedAccumulator, expectedMarks, 6, expectedTouched, 2, 1,
        )
        val actualCount = SparseWorkspaceComparators.scatterAxpyOneMkl(
            external, 0.75, indices, 1, values, 1, 4,
            actualAccumulator, actualMarks, 6, actualTouched, 2, 1,
        )
        assertEquals(expectedCount, actualCount)
        assertContentEquals(expectedAccumulator, actualAccumulator)
        assertContentEquals(expectedMarks, actualMarks)
        assertContentEquals(expectedTouched, actualTouched)
    }

    @Test
    fun `general symmetric product rank and addition operations agree with scalar oracle`() {
        val external = requiredOneMkl() ?: return
        for (transpose in booleanArrayOf(false, true)) {
            val a = sparseComparisonMatrix(if (transpose) 5 else 7, if (transpose) 7 else 5, 0.35, "skewed", benchRng())
            val x = randomVector(5, benchRng())
            val expected = DoubleArray(7) { -0.25 }
            val actual = expected.copyOf()
            scalar.sparseBlas.gemv(0.8, a, x, -0.5, expected, transpose)
            external.prepare(a).use { it.gemv(0.8, x, -0.5, actual, transpose) }
            assertVectorNear(expected, actual, "gemv transpose=$transpose")
        }

        for (lower in booleanArrayOf(false, true)) {
            val a = symmetricTriangle(9, lower)
            val x = randomVector(9, benchRng())
            val expectedVector = DoubleArray(9) { 0.4 }
            val actualVector = expectedVector.copyOf()
            scalar.sparseBlas.symv(0.7, a, x, -0.2, expectedVector, lower)
            external.prepare(a, symmetric = true, lower = lower).use {
                it.symv(0.7, x, -0.2, actualVector)
                it.symv(0.7, x, -0.2, actualVector)
            }
            val repeatedExpected = DoubleArray(9) { 0.4 }
            scalar.sparseBlas.symv(0.7, a, x, -0.2, repeatedExpected, lower)
            scalar.sparseBlas.symv(0.7, a, x, -0.2, repeatedExpected, lower)
            assertVectorNear(repeatedExpected, actualVector, "symv lower=$lower repeated")

            val b = randomMatrix(9, 3, benchRng())
            val expectedMatrix = randomMatrix(9, 3, benchRng())
            val actualMatrix = DenseMatrix.wrap(9, 3, expectedMatrix.data.copyOf())
            scalar.sparseBlas.symm(0.6, a, b, -0.3, expectedMatrix, lower)
            external.prepare(a, symmetric = true, lower = lower).use { it.symm(0.6, b, -0.3, actualMatrix) }
            assertMatrixNear(expectedMatrix, actualMatrix, "symm lower=$lower")
        }

        for (transposeA in booleanArrayOf(false, true)) for (transposeB in booleanArrayOf(false, true)) {
            val a = sparseComparisonMatrix(if (transposeA) 5 else 7, if (transposeA) 7 else 5, 0.3, "banded", benchRng())
            val b = sparseComparisonMatrix(if (transposeB) 6 else 5, if (transposeB) 5 else 6, 0.3, "skewed", benchRng())
            val expected = randomMatrix(7, 6, benchRng())
            val actual = DenseMatrix.wrap(7, 6, expected.data.copyOf())
            scalar.sparseBlas.gemm(0.75, a, transposeA, b, transposeB, -0.4, expected)
            external.denseProduct(0.75, a, transposeA, b, transposeB, -0.4, actual)
            assertMatrixNear(expected, actual, "sp2md transposeA=$transposeA transposeB=$transposeB")
        }

        val productLeft = sparseComparisonMatrix(8, 5, 0.35, "regular", benchRng())
        val productRight = sparseComparisonMatrix(5, 6, 0.35, "skewed", benchRng())
        assertSparseNumericallyNear(
            scalar.sparseBlas.gemm(productLeft, productRight),
            external.sparseProduct(productLeft, productRight),
            "spmm",
        )

        for (transpose in booleanArrayOf(false, true)) {
            val a = sparseComparisonMatrix(7, 5, 0.4, "regular", benchRng())
            val order = if (transpose) a.cols else a.rows
            val expectedDense = randomMatrix(order, order, benchRng())
            val actualDense = DenseMatrix.wrap(order, order, expectedDense.data.copyOf())
            scalar.sparseBlas.syrk(0.8, a, transpose, -0.25, expectedDense, lower = false)
            external.syrkd(0.8, a, transpose, -0.25, actualDense)
            assertTriangleNear(expectedDense, actualDense, lower = false, "syrkd transpose=$transpose")
            assertSparseNumericallyNear(
                scalar.sparseBlas.syrk(a, transpose, lower = false),
                external.syrk(a, transpose),
                "syrk transpose=$transpose",
            )
        }

        for (transpose in booleanArrayOf(false, true)) {
            val a = sparseComparisonMatrix(if (transpose) 5 else 7, if (transpose) 7 else 5, 0.35, "regular", benchRng())
            val b = sparseComparisonMatrix(7, 5, 0.25, "skewed", benchRng())
            assertSparseNumericallyNear(
                scalar.sparseBlas.addScaled(-0.75, a, transpose, b),
                external.addScaled(-0.75, a, transpose, b),
                "add transpose=$transpose",
            )
        }
    }

    @Test
    fun `triangular descriptors orientations and repeated prepared lifetimes agree with scalar oracle`() {
        val external = requiredOneMkl() ?: return
        for (lower in booleanArrayOf(false, true)) for (transpose in booleanArrayOf(false, true)) {
            for (unitDiag in booleanArrayOf(false, true)) {
                val triangle = triangularMatrix(8, lower)
                val rhs = randomVector(8, benchRng())
                val expectedMultiply = rhs.copyOf()
                scalar.sparseBlas.trmv(triangle, expectedMultiply, lower, transpose, unitDiag)
                val actualMultiply = DoubleArray(8)
                external.prepare(triangle, triangular = true, lower = lower, unitDiag = unitDiag).use {
                    it.trmv(rhs, actualMultiply, transpose)
                }
                assertVectorNear(expectedMultiply, actualMultiply, "trmv l=$lower t=$transpose u=$unitDiag")

                val expectedSolve = rhs.copyOf()
                scalar.sparseBlas.trsv(triangle, expectedSolve, lower, transpose, unitDiag)
                val actualSolve = DoubleArray(8)
                external.prepare(triangle, triangular = true, lower = lower, unitDiag = unitDiag).use {
                    it.trsv(rhs, actualSolve, transpose)
                    it.trsv(rhs, actualSolve, transpose)
                }
                assertVectorNear(expectedSolve, actualSolve, "trsv l=$lower t=$transpose u=$unitDiag")

                val dense = randomMatrix(8, 4, benchRng())
                val expectedDenseMultiply = DenseMatrix.wrap(8, 4, dense.data.copyOf())
                scalar.sparseBlas.trmm(triangle, expectedDenseMultiply, lower, transpose, unitDiag)
                val actualDenseMultiply = DenseMatrix.zero(8, 4)
                external.prepare(triangle, triangular = true, lower = lower, unitDiag = unitDiag).use {
                    it.trmm(dense, actualDenseMultiply, transpose)
                }
                assertMatrixNear(expectedDenseMultiply, actualDenseMultiply, "trmm l=$lower t=$transpose u=$unitDiag")

                val expectedDenseSolve = DenseMatrix.wrap(8, 4, dense.data.copyOf())
                scalar.sparseBlas.trsm(triangle, expectedDenseSolve, lower, transpose, unitDiag)
                val actualDenseSolve = DenseMatrix.zero(8, 4)
                external.prepare(triangle, triangular = true, lower = lower, unitDiag = unitDiag).use {
                    it.trsm(dense, actualDenseSolve, transpose)
                }
                assertMatrixNear(expectedDenseSolve, actualDenseSolve, "trsm l=$lower t=$transpose u=$unitDiag")
            }
        }
    }

    private fun requiredOneMkl(): SparseComparator? {
        val comparator = oneMklSparseComparator()
        if (System.getProperty("koblas.oneMklTests") == "true") {
            return checkNotNull(comparator) { "-Pkoblas.oneMklTests=true requires a loader-visible oneMKL runtime" }
        }
        return comparator
    }

    private fun symmetricTriangle(n: Int, lower: Boolean): SparseMatrix = SparseMatrix.ofColumns(
        n,
        n,
        List(n) { j ->
            buildList {
                add(j to (n + 2.0))
                for (i in 0 until n) if (i != j && (if (lower) i > j else i < j) && (i + j) % 3 == 0) {
                    add(i to ((i - j) * 0.125))
                }
            }
        },
    )

    private fun triangularMatrix(n: Int, lower: Boolean): SparseMatrix = SparseMatrix.ofColumns(
        n,
        n,
        List(n) { j ->
            buildList {
                add(j to (2.0 + j * 0.1))
                for (i in 0 until n) if (i != j && (if (lower) i > j else i < j) && (i + j) % 2 == 0) {
                    add(i to ((i + j + 1) * 0.03))
                }
            }
        },
    )

    private fun assertVectorNear(expected: DoubleArray, actual: DoubleArray, context: String) {
        assertEquals(expected.size, actual.size, "$context size")
        for (i in expected.indices) assertEquals(expected[i], actual[i], 1e-10, "$context entry $i")
    }

    private fun assertMatrixNear(expected: DenseMatrix, actual: DenseMatrix, context: String) {
        assertEquals(expected.rows, actual.rows, "$context rows")
        assertEquals(expected.cols, actual.cols, "$context cols")
        for (j in 0 until expected.cols) for (i in 0 until expected.rows) {
            assertEquals(expected[i, j], actual[i, j], 1e-10, "$context entry $i $j")
        }
    }

    private fun assertTriangleNear(expected: DenseMatrix, actual: DenseMatrix, lower: Boolean, context: String) {
        for (j in 0 until expected.cols) for (i in 0 until expected.rows) if (if (lower) i >= j else i <= j) {
            assertEquals(expected[i, j], actual[i, j], 1e-10, "$context entry $i $j")
        }
    }

    private fun assertSparseNumericallyNear(expected: SparseMatrix, actual: SparseMatrix, context: String) {
        assertEquals(expected.rows, actual.rows, "$context rows")
        assertEquals(expected.cols, actual.cols, "$context cols")
        for (j in 0 until expected.cols) for (i in 0 until expected.rows) {
            assertEquals(expected[i, j], actual[i, j], 1e-10, "$context entry $i $j")
        }
    }
}
