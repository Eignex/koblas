package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.poisonedTriangle
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals

class PackedTriangularSolveTest {
    @Test
    fun `packed trsm normalizes every BLAS variant`() {
        val rng = Random(20260912)
        for (order in intArrayOf(1, 3, 4, 5, 9)) {
            for (other in intArrayOf(1, 3, 5)) {
                for (lower in booleanArrayOf(false, true)) {
                    for (transpose in booleanArrayOf(false, true)) {
                        for (unitDiag in booleanArrayOf(false, true)) {
                            for (right in booleanArrayOf(false, true)) {
                                val (triangle, explicit) = poisonedTriangle(rng, order, lower, unitDiag)
                                val expected = if (right) {
                                    randomMatrix(other, order, rng)
                                } else {
                                    randomMatrix(order, other, rng)
                                }
                                val rightHandSide = multiply(explicit, expected, transpose, right)

                                packedTrsmCore(
                                    PortablePackedKernels,
                                    triangle,
                                    rightHandSide,
                                    lower,
                                    transpose,
                                    unitDiag,
                                    right,
                                    Workspace(),
                                )

                                assertClose(
                                    expected,
                                    rightHandSide,
                                    "n=$order other=$other lower=$lower transpose=$transpose " +
                                        "unit=$unitDiag right=$right",
                                    tolerance = 1e-9,
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `ordinary left solve preserves singular zero pivot semantics`() {
        val order = 16
        for (families in listOf(scalarDenseKernelFamilies, platformDenseKernelFamilies)) {
            for (lower in booleanArrayOf(false, true)) {
                val triangle = DenseMatrix.zero(order)
                for (column in 0 until order) {
                    for (row in 0 until order) {
                        triangle[row, column] = when {
                            row == column -> 0.0
                            if (lower) row > column else row < column -> 1.0
                            else -> Double.NaN
                        }
                    }
                }
                assertWideTrsmAgreesWithReference(
                    families,
                    triangle,
                    DenseMatrix.zero(order, 32),
                    lower,
                    transpose = false,
                    right = false,
                )
            }
        }
    }

    @Test
    fun `ordinary solve preserves overflow and cancellation semantics`() {
        val order = 16
        for (families in listOf(scalarDenseKernelFamilies, platformDenseKernelFamilies)) {
            for (lower in booleanArrayOf(false, true)) {
                for (right in booleanArrayOf(false, true)) {
                    val triangle = DenseMatrix.zero(order)
                    val rhs = DoubleArray(order)
                    fun index(i: Int): Int = if (lower) i else order - 1 - i
                    for (column in 0 until order) {
                        for (row in 0 until order) {
                            triangle[index(row), index(column)] = when {
                                row == column -> 1.0
                                row > column -> 1e-300
                                else -> Double.NaN
                            }
                        }
                    }
                    triangle[index(14), index(13)] = 1.0
                    triangle[index(15), index(13)] = 1.0
                    for (sign in doubleArrayOf(-1.0, 1.0)) {
                        rhs[index(13)] = sign * 1e308
                        rhs[index(14)] = sign * 1e308
                        rhs[index(15)] = 1e308
                        val source = if (right) DenseMatrix.zero(32, order) else DenseMatrix.zero(order, 32)
                        for (panel in 0 until 32) {
                            for (i in 0 until order) {
                                if (right) source[panel, i] = rhs[i] else source[i, panel] = rhs[i]
                            }
                        }
                        assertWideTrsmAgreesWithReference(families, triangle, source, lower, transpose = !right, right)
                    }
                }
            }
        }
    }

    @Suppress("LongParameterList")
    private fun assertWideTrsmAgreesWithReference(
        families: DenseKernelFamilies,
        triangle: DenseMatrix,
        source: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        right: Boolean,
    ) {
        val reference = BuiltinBlas(families)
        val order = triangle.rows
        val panels = if (right) source.rows else source.cols
        val expected = DenseMatrix.wrap(source.rows, source.cols, source.data.copyOf())
        for (panel in 0 until panels) {
            // One RHS retains the reference walk independently of the wide call's packed dispatch.
            val rhs = DoubleArray(order) { i -> if (right) source[panel, i] else source[i, panel] }
            val narrow = if (right) DenseMatrix.wrap(1, order, rhs) else DenseMatrix.wrap(order, 1, rhs)
            reference.trsm(triangle, narrow, lower, transpose, right = right)
            for (i in 0 until order) {
                if (right) expected[panel, i] = rhs[i] else expected[i, panel] = rhs[i]
            }
        }
        val actual = DenseMatrix.wrap(source.rows, source.cols, source.data.copyOf())

        reference.trsm(triangle, actual, lower, transpose, right = right)

        assertContentEquals(
            expected.data,
            actual.data,
            "${families.vector.name} lower=$lower transpose=$transpose right=$right",
        )
    }

    @Suppress("LongParameterList")
    private fun multiply(triangle: DenseMatrix, x: DenseMatrix, transpose: Boolean, right: Boolean): DenseMatrix {
        val result = DenseMatrix.zero(x.rows, x.cols)
        if (right) {
            for (column in 0 until x.cols) {
                for (inner in 0 until x.cols) {
                    val coefficient = if (transpose) triangle[column, inner] else triangle[inner, column]
                    for (row in 0 until x.rows) result[row, column] += x[row, inner] * coefficient
                }
            }
        } else {
            for (column in 0 until x.cols) {
                for (inner in 0 until x.rows) {
                    for (row in 0 until x.rows) {
                        val coefficient = if (transpose) triangle[inner, row] else triangle[row, inner]
                        result[row, column] += coefficient * x[inner, column]
                    }
                }
            }
        }
        return result
    }
}
