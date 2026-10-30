package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.ReferenceBlas
import kotlin.test.Test
import kotlin.test.assertContentEquals

class SparseMatrixViewTest {
    @Test
    fun `sparse pair products leave empty padded destinations untouched`() {
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()) {
            for ((rows, columns) in listOf(0 to 3, 3 to 0)) {
                for (size in intArrayOf(0, 3)) {
                    for (alpha in doubleArrayOf(0.0, 1.0)) {
                        for (beta in doubleArrayOf(0.0, 0.5, 1.0)) {
                            val a = SparseMatrix.ofColumns(rows, 1, listOf(emptyList()))
                            val b = SparseMatrix.ofColumns(1, columns, List(columns) { emptyList() })
                            val backing = DoubleArray(size) { -77.0 }
                            val before = backing.copyOf()
                            val c = DenseMatrixView.wrap(rows, columns, backing, size, rows + 1)

                            engine.gemm(alpha, a, false, b, false, beta, c)

                            assertContentEquals(before, backing)
                        }
                    }
                }
            }
        }
    }

    private fun region(seed: Int): DenseMatrixView {
        val parent = DenseMatrix.wrap(9, 8, DoubleArray(72) { -77.0 })
        val result = parent.view(1, 1, 7, 6).view(1, 1, 5, 5)
        for (j in 0 until 5) for (i in 0 until 5) result[i, j] = (i + j + seed) / 17.0
        return result
    }

    private fun copy(a: Matrix): DenseMatrix = DenseMatrix.ofRows(a.toArray())

    private fun source(): SparseMatrix = SparseMatrix.ofColumns(
        5,
        5,
        List(5) { j ->
            List(5) { i -> i to if (i == j) 4.0 else (i + j + 1) / 31.0 }
        },
    )

    private fun assertRegionAgreesWithReference(expected: DenseMatrix, actual: DenseMatrixRegion, before: DoubleArray) {
        for (j in 0 until actual.cols) {
            for (i in 0 until actual.rows) {
                before[actual.offset + i + j * actual.leadingDimension] = expected[i, j]
            }
        }
        assertClose(before, actual.values, "sparse view backing")
    }

    @Test
    fun `mixed and generic products retain dense operand addressing`() {
        val sparse = source()
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()) {
            for (transposeA in booleanArrayOf(false, true)) {
                for (transposeB in booleanArrayOf(false, true)) {
                    for (right in booleanArrayOf(false, true)) {
                        val b = region(2)
                        val c = region(3)
                        val expected = copy(c)
                        val before = c.values.copyOf()
                        ReferenceSparseBlas.gemm(0.875, sparse, transposeA, copy(b), transposeB, -0.25, expected, right)
                        val generic: Matrix = sparse

                        if (right) {
                            engine.gemm(0.875, b as Matrix, transposeB, generic, transposeA, -0.25, c)
                        } else {
                            engine.gemm(0.875, generic, transposeA, b as Matrix, transposeB, -0.25, c)
                        }

                        assertRegionAgreesWithReference(expected, c, before)
                    }
                    val c = region(4)
                    val expected = copy(c)
                    val before = c.values.copyOf()
                    ReferenceSparseBlas.gemm(0.875, sparse, transposeA, sparse, transposeB, -0.25, expected)

                    engine.gemm(
                        0.875,
                        sparse as Matrix,
                        transposeA,
                        sparse as Matrix,
                        transposeB,
                        -0.25,
                        c,
                    )

                    assertRegionAgreesWithReference(expected, c, before)
                }
            }
        }
    }

    @Test
    fun `sparse structured routines preserve parent entries`() {
        val a = source()
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()) {
            for (lower in booleanArrayOf(false, true)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (right in booleanArrayOf(false, true)) {
                        for (operation in listOf("symm", "syrk", "trsm", "trmm")) {
                            val b = region(2)
                            val c = region(3)
                            val before = c.values.copyOf()
                            val expected = copy(c)
                            when (operation) {
                                "symm" -> {
                                    ReferenceSparseBlas.symm(0.875, a, copy(b), -0.25, expected, lower, right)
                                    engine.symm(0.875, a, b, -0.25, c, lower, right)
                                }

                                "syrk" -> {
                                    ReferenceSparseBlas.syrk(0.875, a, transpose, -0.25, expected, lower)
                                    engine.syrk(0.875, a, transpose, -0.25, c, lower)
                                }

                                "trsm" -> {
                                    ReferenceBlas.trsm(copy(a), expected, lower, transpose, true, right, 0.875)
                                    engine.trsm(a as Matrix, c, lower, transpose, true, right, 0.875)
                                }

                                else -> {
                                    ReferenceBlas.trmm(copy(a), expected, lower, transpose, true, right, 0.875)
                                    engine.trmm(a as Matrix, c, lower, transpose, true, right, 0.875)
                                }
                            }
                            assertRegionAgreesWithReference(expected, c, before)
                        }
                    }
                }
            }
        }
    }
}
