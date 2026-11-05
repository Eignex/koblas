package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.dense.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Regions of one backing array: an operand overlapping the destination is read as it was before the call,
 * and one that only shares the array is read in place.
 */
class RegionPreparationTest {
    // The destination is a row or a column of the matrix operand, and also the vector operand when that one
    // is stored the same way round, so both reads have to see the values from before the call.
    @Test
    fun `single row and column products snapshot operands holding the destination`() = withRegionEngines { blas ->
        for (row in booleanArrayOf(true, false)) {
            for (transposed in booleanArrayOf(false, true)) {
                val parent = DenseMatrix.wrap(6, 6, DoubleArray(36) { (it % 13 - 6) / 16.0 })
                val c = if (row) parent.view(2, 0, 1, 6) else parent.view(0, 2, 6, 1)
                val vector = when {
                    !transposed -> c
                    row -> paddedRegion(6, 1)
                    else -> paddedRegion(1, 6)
                }
                val a = if (row) vector else parent
                val b = if (row) parent else vector
                val ta = row && transposed
                val tb = !row && transposed
                val before = parent.values.copyOf()
                val expected = contiguousCopy(c)
                ReferenceBlas.gemm(0.875, contiguousCopy(a), ta, contiguousCopy(b), tb, -0.25, expected)

                blas.gemm(0.875, a, ta, b, tb, -0.25, c, scratch = Scratch())

                assertRegionAgreesWithReference(expected, c, before, "row=$row transposed=$transposed")
            }
        }
    }

    @Test
    fun `overlapping rectangles snapshot the pre call operands`() = withRegionEngines { blas ->
        for (ta in booleanArrayOf(false, true)) {
            for (tb in booleanArrayOf(false, true)) {
                val parent = DenseMatrix.wrap(8, 7, DoubleArray(56) { (it % 17 - 8) / 16.0 })
                val a = parent.view(1, 1, 4, 4)
                val b = parent.view(2, 1, 4, 4)
                val c = parent.view(2, 2, 4, 4)
                val before = parent.values.copyOf()
                val expected = contiguousCopy(c)
                ReferenceBlas.gemm(0.875, contiguousCopy(a), ta, contiguousCopy(b), tb, -0.25, expected)

                blas.gemm(0.875, a, ta, b, tb, -0.25, c, scratch = Scratch())

                assertRegionAgreesWithReference(expected, c, before, "tA=$ta tB=$tb")
            }
        }
    }

    @Test
    fun `disjoint trailing updates borrow no panel snapshot`() = withEngines { blas ->
        val parent = paddedRegion(12, 12)
        val a = parent.view(4, 0, 8, 4)
        val c = parent.view(4, 4, 8, 8)
        val expected = contiguousCopy(c)
        val before = parent.values.copyOf()
        val workspace = Scratch()
        ReferenceBlas.syrk(-1.0, contiguousCopy(a), false, 1.0, expected)

        blas.syrk(-1.0, a, false, 1.0, c, scratch = workspace)

        assertRegionAgreesWithReference(expected, c, before)
        assertEquals(0, workspace.available(a.rows * a.cols))
    }

    @Test
    fun `disjoint rectangles of one parent agree with the reference`() = withRegionEngines { blas ->
        val parent = paddedRegion(12, 12)
        val panel = parent.view(4, 0, 8, 4)
        val beside = parent.view(0, 4, 4, 8)
        val triangle = parent.view(0, 0, 4, 4)
        val trailing = parent.view(4, 4, 8, 8)
        val expectedTrailing = contiguousCopy(trailing)
        val expectedBeside = contiguousCopy(beside)
        ReferenceBlas.syrk(-1.0, contiguousCopy(panel), false, 1.0, expectedTrailing)
        ReferenceBlas.gemm(0.5, contiguousCopy(panel), false, contiguousCopy(beside), false, 1.0, expectedTrailing)
        ReferenceBlas.trsm(contiguousCopy(triangle), expectedBeside, lower = true)
        val before = parent.values.copyOf()

        blas.syrk(-1.0, panel, false, 1.0, trailing, scratch = Scratch())
        blas.gemm(0.5, panel, false, beside, false, 1.0, trailing, scratch = Scratch())
        assertRegionAgreesWithReference(expectedTrailing, trailing, before, "trailing")
        val afterProducts = parent.values.copyOf()
        blas.trsm(triangle, beside, lower = true, scratch = Scratch())

        assertRegionAgreesWithReference(expectedBeside, beside, afterProducts, "beside")
    }

    @Test
    fun `borrowed triangular operands snapshot overlapping destinations`() = withRegionEngines { blas ->
        for (lower in booleanArrayOf(false, true)) {
            for (right in booleanArrayOf(false, true)) {
                val parent = paddedRegion(7, 7)
                val a = parent.view(0, 0, 4, 4)
                val b = parent.view(1, 1, 4, 4)
                val expected = contiguousCopy(b)
                val before = parent.values.copyOf()
                ReferenceBlas.trsm(contiguousCopy(a), expected, lower, right = right)

                blas.trsm(a, b, lower, right = right, scratch = Scratch())

                assertRegionAgreesWithReference(expected, b, before, "lower=$lower right=$right")
            }
        }
    }

    @Test
    fun `borrowed rank updates snapshot overlapping inputs`() = withRegionEngines { blas ->
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                val parent = paddedRegion(7, 7)
                val a = parent.view(0, 0, if (transpose) 3 else 4, if (transpose) 4 else 3)
                val c = parent.view(1, 1, 4, 4)
                val expected = contiguousCopy(c)
                val before = parent.values.copyOf()
                ReferenceBlas.syrk(-1.0, contiguousCopy(a), transpose, 1.0, expected, lower)

                blas.syrk(-1.0, a, transpose, 1.0, c, lower, scratch = Scratch())

                assertRegionAgreesWithReference(expected, c, before, "transpose=$transpose lower=$lower")
            }
        }
    }

    @Test
    fun `matrix vector products snapshot a shared destination buffer`() = withRegionEngines { blas ->
        val actual = doubleArrayOf(2.0, -0.5, 4.0)
        val a = DenseMatrixView.wrap(3, 1, actual, leadingDimension = 5)
        val expected = actual.copyOf()
        ReferenceBlas.gemv(0.875, contiguousCopy(a), doubleArrayOf(2.0), -0.25, expected)

        blas.gemv(0.875, a, doubleArrayOf(2.0), 0, 1, -0.25, actual, 0, 1, scratch = Scratch())

        assertClose(expected, actual, "aliased region gemv")
    }

    // An order inside one diagonal block reaches nothing but the substitution, so the staged triangle is the
    // whole of what the workspace lent and a second length would be scratch the call had no use for.
    @Test
    fun `a triangular solve stages its triangle from the workspace and borrows nothing else`() {
        val rng = Random(20260929)
        val n = 5
        val workspace = Scratch()
        val start = randomMatrix(n, n, rng)
        for (i in 0 until n) start[i, i] = 3.0 + i % 2
        val expected = start.copyOf()
        ReferenceBlas.trsm(start.copyOf(), expected, lower = true)
        val aliased = start.copyOf()

        KoblasEngine.scalar.trsm(aliased, aliased, lower = true, scratch = workspace)

        assertClose(expected.values, aliased.values, "trsm with its own triangle", tolerance = 1e-9)
        assertEquals(1, workspace.available(n * n), "the triangle was not staged from the workspace")
        assertEquals(1, workspace.idleLengths(), "the solve borrowed scratch besides its staged triangle")
    }

    @Test
    fun `a repeated call over one shape reuses the same scratch`() {
        val rng = Random(20260930)
        val order = 9
        val workspace = Scratch()
        val a = randomMatrix(order, order, rng)
        val b = randomMatrix(order, 4, rng)
        val blas = KoblasEngine.scalar

        blas.gemm(0.5, a, false, b, false, 0.5, b, scratch = workspace)
        val afterOne = workspace.idleLengths()
        repeat(3) { blas.gemm(0.5, a, false, b, false, 0.5, b, scratch = workspace) }

        assertTrue(afterOne > 0, "an aliased product needs an operand snapshot")
        assertEquals(afterOne, workspace.idleLengths(), "a repeated call kept asking for new buffers")
    }
}
