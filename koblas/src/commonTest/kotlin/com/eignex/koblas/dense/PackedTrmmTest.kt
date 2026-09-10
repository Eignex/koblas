package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.poisonedTriangle
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class PackedTrmmTest {
    @Test
    fun `packed multiply agrees with a written out product for every flag combination`() {
        val rng = Random(202609091)
        val shapes = listOf(3 to 5, 7 to 3, 16 to 9, 17 to 33)
        for ((order, panel) in shapes) {
            for (right in booleanArrayOf(false, true)) {
                for (lower in booleanArrayOf(false, true)) {
                    for (transpose in booleanArrayOf(false, true)) {
                        for (unitDiagonal in booleanArrayOf(false, true)) {
                            val (triangle, explicit) = poisonedTriangle(rng, order, lower, unitDiagonal)
                            val source = if (right) randomMatrix(panel, order, rng) else randomMatrix(order, panel, rng)
                            val expected = writtenOut(explicit, source, transpose, right)
                            val actual = DenseMatrix(source.rows, source.cols, source.data.copyOf())

                            packedTrmmCore(
                                platformDenseKernelFamilies.packed,
                                triangle,
                                actual,
                                lower,
                                transpose,
                                unitDiagonal,
                                right,
                                Workspace(),
                            )

                            assertClose(
                                expected,
                                actual,
                                "order=$order panel=$panel right=$right lower=$lower " +
                                    "transpose=$transpose unit=$unitDiagonal",
                                tolerance = 1e-11,
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `packed multiply crosses tile and cache block boundaries`() {
        val rng = Random(202609092)
        val shapes = listOf(
            31 to 7,
            32 to 8,
            33 to 9,
            DenseTuning.packedBlockDepth - 1 to 3,
            DenseTuning.packedBlockDepth to 4,
            DenseTuning.packedBlockDepth + 1 to 5,
        )
        for ((order, panel) in shapes) {
            val (triangle, explicit) = poisonedTriangle(rng, order, lower = true, unitDiag = true)
            val source = randomMatrix(panel, order, rng)
            val expected = writtenOut(explicit, source, transpose = true, right = true)
            val actual = DenseMatrix(source.rows, source.cols, source.data.copyOf())

            packedTrmmCore(
                PortablePackedKernels,
                triangle,
                actual,
                lower = true,
                transpose = true,
                unitDiagonal = true,
                right = true,
                workspace = Workspace(),
            )

            assertClose(expected, actual, "order=$order panel=$panel", tolerance = 1e-10)
        }
    }

    @Test
    fun `ordinary multiply applies alpha before the packed product`() {
        val rng = Random(202609093)
        val order = DenseTuning.trmmPackedMinOrder
        val panel = DenseTuning.trmmPackedMinRows
        val alpha = -0.75
        val (triangle, explicit) = poisonedTriangle(rng, order, lower = false, unitDiag = true)
        val source = randomMatrix(order, panel, rng)
        val scaled = DenseMatrix(source.rows, source.cols, DoubleArray(source.data.size) { alpha * source.data[it] })
        val expected = writtenOut(explicit, scaled, transpose = true, right = false)
        val actual = DenseMatrix(source.rows, source.cols, source.data.copyOf())

        BuiltinBlas(platformDenseKernelFamilies).trmm(
            triangle,
            actual,
            lower = false,
            transpose = true,
            unitDiag = true,
            alpha = alpha,
            workspace = Workspace(),
        )

        assertClose(expected, actual, "alpha", tolerance = 1e-11)
    }

    @Test
    fun `packed multiply snapshots a triangle that shares the result backing`() {
        val rng = Random(202609094)
        val order = maxOf(DenseTuning.trmmPackedMinOrder, DenseTuning.trmmPackedMinRows)
        val shared = randomMatrix(order, order, rng)
        val sourceSnapshot = DenseMatrix(order, order, shared.data.copyOf())
        val explicitTriangle = DenseMatrix.zero(order, order)
        for (column in 0 until order) {
            for (row in column until order) explicitTriangle[row, column] = shared[row, column]
        }
        val expected = writtenOut(explicitTriangle, sourceSnapshot, transpose = false, right = true)

        BuiltinBlas(platformDenseKernelFamilies).trmm(
            shared,
            shared,
            lower = true,
            right = true,
            workspace = Workspace(),
        )

        assertClose(expected, shared, "shared backing", tolerance = 1e-10)
    }

    @Test
    fun `ordinary multiply preserves exceptional zero arithmetic`() {
        val order = DenseTuning.trmmPackedMinOrder
        val panel = DenseTuning.trmmPackedMinRows
        val triangle = DenseMatrix.diagonal(order)
        val source = DenseMatrix(panel, order)
        source[0, 0] = Double.POSITIVE_INFINITY
        source[0, 1] = 1.0

        BuiltinBlas(platformDenseKernelFamilies).trmm(
            triangle,
            source,
            lower = true,
            right = true,
            workspace = Workspace(),
        )

        assertEquals(Double.POSITIVE_INFINITY, source[0, 0])
        assertEquals(1.0, source[0, 1])
    }

    @Test
    fun `ordinary multiply preserves overflow and cancellation`() {
        val order = DenseTuning.trmmPackedMinOrder
        val triangle = DenseMatrix.diagonal(order)
        triangle[0, 0] = 2.0
        triangle[1, 0] = -2.0
        val source = DenseMatrix(order, DenseTuning.trmmPackedMinRows)
        source[0, 0] = Double.MAX_VALUE
        source[1, 0] = Double.MAX_VALUE

        BuiltinBlas(platformDenseKernelFamilies).trmm(
            triangle,
            source,
            lower = true,
            workspace = Workspace(),
        )

        assertEquals(Double.POSITIVE_INFINITY, source[0, 0])
        assertEquals(Double.NEGATIVE_INFINITY, source[1, 0])
    }

    private fun writtenOut(
        triangle: DenseMatrix,
        source: DenseMatrix,
        transpose: Boolean,
        right: Boolean,
    ): DenseMatrix {
        val result = DenseMatrix(source.rows, source.cols)
        if (right) {
            for (column in 0 until source.cols) {
                for (row in 0 until source.rows) {
                    var sum = 0.0
                    for (step in 0 until source.cols) {
                        val coefficient = if (transpose) triangle[column, step] else triangle[step, column]
                        sum += source[row, step] * coefficient
                    }
                    result[row, column] = sum
                }
            }
        } else {
            for (column in 0 until source.cols) {
                for (row in 0 until source.rows) {
                    var sum = 0.0
                    for (step in 0 until source.rows) {
                        val coefficient = if (transpose) triangle[step, row] else triangle[row, step]
                        sum += coefficient * source[step, column]
                    }
                    result[row, column] = sum
                }
            }
        }
        return result
    }
}
