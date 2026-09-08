package com.eignex.koblas.dense

import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.poisonedTriangle
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

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
                                    ScalarKernels,
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
                                    "n=$order other=$other lower=$lower transpose=$transpose unit=$unitDiag right=$right",
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
    fun `ordinary trsm reaches packed update and solve kernels`() {
        var solves = 0
        var fused = 0
        val recording = object : Kernels by ScalarKernels {
            override fun trsmTile(
                validRows: Int,
                order: Int,
                packedTriangle: DoubleArray,
                triangleOff: Int,
                lower: Boolean,
                unitDiag: Boolean,
                x: DoubleArray,
                xOff: Int,
            ) {
                solves++
                ScalarKernels.trsmTile(
                    validRows, order, packedTriangle, triangleOff, lower, unitDiag, x, xOff,
                )
            }

            override fun gemmTrsmTile(
                depth: Int,
                validRows: Int,
                order: Int,
                packedA: DoubleArray,
                aOff: Int,
                packedB: DoubleArray,
                bOff: Int,
                packedTriangle: DoubleArray,
                triangleOff: Int,
                lower: Boolean,
                unitDiag: Boolean,
                x: DoubleArray,
                xOff: Int,
            ) {
                fused++
                ScalarKernels.gemmTrsmTile(
                    depth, validRows, order, packedA, aOff, packedB, bOff,
                    packedTriangle, triangleOff, lower, unitDiag, x, xOff,
                )
            }
        }
        val order = maxOf(DenseTuning.trsmPackedMinOrder, 17)
        val rows = maxOf(DenseTuning.trsmPackedMinRows, 9)
        val rng = Random(20260913)
        val (triangle, explicit) = poisonedTriangle(rng, order, lower = true, unitDiag = false)
        val expected = randomMatrix(rows, order, rng)
        val rightHandSide = multiply(explicit, expected, transpose = true, right = true)

        triangularMatrix(
            recording, triangle, rightHandSide,
            lower = true, transpose = true, unitDiag = false, right = true,
            alpha = 1.0, solve = true, workspace = Workspace(),
        )

        assertClose(expected, rightHandSide, "ordinary packed right transposed solve", tolerance = 1e-9)
        assertTrue(solves > 0, "ordinary trsm did not use trsmTile")
        assertTrue(fused > 0, "ordinary trsm did not use gemmTrsmTile")
    }

    @Suppress("LongParameterList")
    private fun multiply(
        triangle: DenseMatrix,
        x: DenseMatrix,
        transpose: Boolean,
        right: Boolean,
    ): DenseMatrix {
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
