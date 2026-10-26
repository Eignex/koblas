package com.eignex.koblas.dense

import com.eignex.koblas.*
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class DenseMatrixViewTest {
    private fun region(rows: Int, cols: Int, seed: Int = 1): DenseMatrixView {
        val parent = DenseMatrix.wrap(rows + 4, cols + 3, DoubleArray((rows + 4) * (cols + 3)) { -999.0 })
        val view = parent.view(1, 1, rows + 2, cols + 1).view(1, 1, rows, cols)
        for (j in 0 until cols) {
            for (i in 0 until rows) {
                view[i, j] = if (i == j) rows + seed.toDouble() else ((i * 3 + j * 7 + seed) % 11 - 5) / 32.0
            }
        }
        return view
    }

    private fun copy(a: Matrix): DenseMatrix = DenseMatrix.wrap(
        a.rows,
        a.cols,
        DoubleArray(a.rows * a.cols) { a[it % a.rows, it / a.rows] },
    )

    private fun assertRegionAgreesWithReference(expected: DenseMatrix, actual: DenseMatrixRegion, before: DoubleArray) {
        val full = before.copyOf()
        val addressed = BooleanArray(full.size)
        for (j in 0 until actual.cols) {
            for (i in 0 until actual.rows) {
                val index = actual.offset + i + j * actual.leadingDimension
                full[index] = expected[i, j]
                addressed[index] = true
            }
        }
        for (index in full.indices) {
            if (!addressed[index] || full[index].isNaN()) {
                assertEquals(full[index], actual.values[index])
            } else {
                assertClose(full[index], actual.values[index], "view backing index $index")
            }
        }
    }

    private fun withRegions(body: (DenseBlas) -> Unit) {
        withDenseBlas(body)
        body(PackedRegionBlas(testDenseBlas))
    }

    @Test
    fun `rectangular products agree across transpose modes and padded destinations`() = withRegions { blas ->
        for (ta in booleanArrayOf(false, true)) {
            for (tb in booleanArrayOf(false, true)) {
                val a = region(if (ta) 5 else 7, if (ta) 7 else 5)
                val b = region(if (tb) 6 else 5, if (tb) 5 else 6, 2)
                val c = region(7, 6, 3)
                val before = c.values.copyOf()
                val expected = copy(c)
                ReferenceBlas.gemm(0.875, copy(a), ta, copy(b), tb, -0.25, expected)

                blas.gemm(0.875, a, ta, b, tb, -0.25, c, Workspace())

                assertRegionAgreesWithReference(expected, c, before)
            }
        }
    }

    @Test
    fun `packed products retain destination padding`() = withRegions { blas ->
        val a = region(67, 65)
        val b = region(65, 69, 2)
        val c = region(67, 69, 3)
        val expected = copy(c)
        val before = c.values.copyOf()
        ReferenceBlas.gemm(1.0, copy(a), false, copy(b), false, 0.0, expected)

        blas.gemm(1.0, a, false, b, false, 0.0, c, Workspace())

        assertRegionAgreesWithReference(expected, c, before)
    }

    @Test
    fun `rank updates preserve the unselected triangle and parent entries`() = withRegions { blas ->
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                val a = region(if (transpose) 4 else 7, if (transpose) 7 else 4)
                val c = region(7, 7)
                val expected = copy(c)
                val before = c.values.copyOf()
                ReferenceBlas.syrk(-1.0, copy(a), transpose, 0.5, expected, lower)

                blas.syrk(-1.0, a, transpose, 0.5, c, lower, Workspace())

                assertRegionAgreesWithReference(expected, c, before)
            }
        }
    }

    @Test
    fun `triangular matrix solves agree across flags and diagonal blocks`() = withRegions { blas ->
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (right in booleanArrayOf(false, true)) {
                    for (unit in booleanArrayOf(false, true)) {
                        val a = region(67, 67)
                        val b = region(if (right) 5 else 67, if (right) 67 else 5, 2)
                        val expected = copy(b)
                        val before = b.values.copyOf()
                        ReferenceBlas.trsm(copy(a), expected, lower, transpose, unit, right, 0.875)

                        blas.trsm(a, b, lower, transpose, unit, right, 0.875, Workspace())

                        assertRegionAgreesWithReference(expected, b, before)
                    }
                }
            }
        }
    }

    @Test
    fun `matrix vector calls address the borrowed rectangle`() = withRegions { blas ->
        for (transpose in booleanArrayOf(false, true)) {
            val a = region(7, 5)
            val x = DoubleArray(if (transpose) 7 else 5) { it / 8.0 }
            val actual = DoubleArray(if (transpose) 5 else 7) { it / 16.0 }
            val expected = actual.copyOf()
            ReferenceBlas.gemv(0.875, copy(a), x, -0.25, expected, transpose)

            blas.gemv(0.875, a, x, -0.25, actual, transpose)

            assertClose(expected, actual, "view vector")
        }
    }

    @Test
    fun `triangular vector solves agree across flags`() = withRegions { blas ->
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (unit in booleanArrayOf(false, true)) {
                    val a = region(7, 7)
                    val actual = DoubleArray(7) { it / 8.0 }
                    val expected = actual.copyOf()
                    ReferenceBlas.trsv(copy(a), expected, lower, transpose, unit)

                    blas.trsv(a, actual, lower, transpose, unit, Workspace())

                    assertClose(expected, actual, "view vector")
                }
            }
        }
    }

    @Test
    fun `overlapping rectangles snapshot the pre call operands`() = withRegions { blas ->
        for (ta in booleanArrayOf(false, true)) {
            for (tb in booleanArrayOf(false, true)) {
                val parent = DenseMatrix.wrap(8, 7, DoubleArray(56) { (it % 17 - 8) / 16.0 })
                val a = parent.view(1, 1, 4, 4)
                val b = parent.view(2, 1, 4, 4)
                val c = parent.view(2, 2, 4, 4)
                val before = parent.values.copyOf()
                val expected = copy(c)
                ReferenceBlas.gemm(0.875, copy(a), ta, copy(b), tb, -0.25, expected)

                blas.gemm(0.875, a, ta, b, tb, -0.25, c, Workspace())

                assertRegionAgreesWithReference(expected, c, before)
            }
        }
    }

    @Test
    fun `disjoint trailing updates borrow no panel snapshot`() = withDenseBlas { blas ->
        val parent = region(12, 12)
        val a = parent.view(4, 0, 8, 4)
        val c = parent.view(4, 4, 8, 8)
        val expected = copy(c)
        val before = parent.values.copyOf()
        val workspace = Workspace()
        ReferenceBlas.syrk(-1.0, copy(a), false, 1.0, expected)

        blas.syrk(-1.0, a, false, 1.0, c, workspace = workspace)

        assertRegionAgreesWithReference(expected, c, before)
        assertEquals(0, workspace.available(a.rows * a.cols))
    }

    @Test
    fun `zero multipliers overwrite only the destination rectangle`() = withRegions { blas ->
        val a = region(4, 4).also { it.values.fill(Double.NaN) }
        val b = region(4, 4).also { it.values.fill(Double.NaN) }
        val c = region(4, 4).also { it.values.fill(Double.NaN) }
        val before = c.values.copyOf()
        val expected = DenseMatrix.zero(4)
        val workspace = Workspace()

        blas.gemm(0.0, a, false, b, false, 0.0, c, workspace)

        assertRegionAgreesWithReference(expected, c, before)
        assertEquals(0, workspace.available(16))
        blas.trsm(a, c, lower = true, alpha = 0.0, workspace = workspace)
        assertRegionAgreesWithReference(expected, c, before)
    }

    @Test
    fun `empty calls validate shapes and leave backing buffers untouched`() = withRegions { blas ->
        val a = region(0, 3)
        val b = region(3, 4)
        val c = region(0, 4)
        val before = c.values.copyOf()

        blas.gemm(1.0, a, false, b, false, 0.0, c)

        assertContentEquals(before, c.values)
        assertFailsWith<DimensionMismatch> { blas.gemm(0.0, a, false, b, false, 0.0, region(1, 4)) }
    }

    @Test
    fun `borrowed triangular operands snapshot overlapping destinations`() = withRegions { blas ->
        for (lower in booleanArrayOf(false, true)) {
            for (right in booleanArrayOf(false, true)) {
                val parent = region(7, 7)
                val a = parent.view(0, 0, 4, 4)
                val b = parent.view(1, 1, 4, 4)
                val expected = copy(b)
                val before = parent.values.copyOf()
                ReferenceBlas.trsm(copy(a), expected, lower, right = right)

                blas.trsm(a, b, lower, right = right, workspace = Workspace())

                assertRegionAgreesWithReference(expected, b, before)
            }
        }
    }

    @Test
    fun `borrowed rank updates snapshot overlapping inputs`() = withRegions { blas ->
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                val parent = region(7, 7)
                val a = parent.view(0, 0, if (transpose) 3 else 4, if (transpose) 4 else 3)
                val c = parent.view(1, 1, 4, 4)
                val expected = copy(c)
                val before = parent.values.copyOf()
                ReferenceBlas.syrk(-1.0, copy(a), transpose, 1.0, expected, lower)

                blas.syrk(-1.0, a, transpose, 1.0, c, lower, Workspace())

                assertRegionAgreesWithReference(expected, c, before)
            }
        }
    }

    @Test
    fun `matrix vector products snapshot a shared destination buffer`() = withRegions { blas ->
        val actual = doubleArrayOf(2.0, -0.5, 4.0)
        val a = DenseMatrixView.wrap(3, 1, actual, leadingDimension = 5)
        val expected = actual.copyOf()
        ReferenceBlas.gemv(0.875, copy(a), doubleArrayOf(2.0), -0.25, expected)

        blas.gemv(0.875, a, doubleArrayOf(2.0), -0.25, actual, workspace = Workspace())

        assertClose(expected, actual, "aliased region gemv")
    }

    @Test
    fun `zero rank and vector multipliers skip poisoned inputs`() = withRegions { blas ->
        val a = region(4, 4).also { it.values.fill(Double.NaN) }
        val c = region(4, 4)
        val before = c.values.copyOf()
        val expected = copy(c)
        for (j in 0 until 4) for (i in j until 4) expected[i, j] = 0.0
        val y = DoubleArray(4) { Double.NaN }

        blas.syrk(0.0, a, false, 0.0, c)
        blas.gemv(0.0, a, DoubleArray(4) { Double.NaN }, 0.0, y)

        assertRegionAgreesWithReference(expected, c, before)
        assertContentEquals(DoubleArray(4), y)
    }

    @Test
    fun `zero beta skips poisoned destination entries`() = withRegions { blas ->
        val a = region(4, 3)
        val b = region(3, 5, 2)
        val c = region(4, 5).also { it.values.fill(Double.NaN) }
        val before = c.values.copyOf()
        val expected = DenseMatrix.zero(4, 5)
        ReferenceBlas.gemm(1.0, copy(a), false, copy(b), false, 0.0, expected)

        blas.gemm(1.0, a, false, b, false, 0.0, c)

        assertRegionAgreesWithReference(expected, c, before)
    }

    @Test
    fun `contiguous and borrowed operands mix in allocating overloads`() = withRegions { blas ->
        val a = region(4, 3)
        val b = copy(region(3, 5, 2))
        val expected = DenseMatrix.zero(4, 5)
        ReferenceBlas.gemm(1.0, copy(a), false, b, false, 0.0, expected)

        val actual = blas.gemm(a, b)

        assertClose(expected, actual, "mixed storage gemm")
        val x = DoubleArray(3) { it / 4.0 }
        val vector = DoubleArray(4)
        ReferenceBlas.gemv(1.0, copy(a), x, 0.0, vector)
        assertClose(vector, blas.gemv(a, x), "allocating view gemv")
    }

    @Test
    fun `empty shared dimensions scale rectangular destinations`() = withRegions { blas ->
        val a = region(4, 0)
        val b = region(0, 3)
        val c = region(4, 3)
        val expected = copy(c)
        val before = c.values.copyOf()
        ReferenceBlas.gemm(1.0, copy(a), false, copy(b), false, 0.5, expected)

        blas.gemm(1.0, a, false, b, false, 0.5, c)

        assertRegionAgreesWithReference(expected, c, before)
    }

    @Test
    fun `rank updates over empty inputs scale the selected triangle`() = withRegions { blas ->
        for (lower in booleanArrayOf(false, true)) {
            val a = region(4, 0)
            val c = region(4, 4)
            val expected = copy(c)
            val before = c.values.copyOf()
            ReferenceBlas.syrk(1.0, copy(a), false, 0.5, expected, lower)

            blas.syrk(1.0, a, false, 0.5, c, lower)

            assertRegionAgreesWithReference(expected, c, before)
        }
    }

    @Test
    fun `matrix vector calls over no columns scale the output`() = withRegions { blas ->
        val a = region(4, 0)
        val actual = doubleArrayOf(1.0, -2.0, 3.0, -4.0)
        val expected = actual.copyOf()
        ReferenceBlas.gemv(1.0, copy(a), doubleArrayOf(), 0.5, expected)

        blas.gemv(1.0, a, doubleArrayOf(), 0.5, actual)

        assertClose(expected, actual, "empty inner dimension gemv")
    }
}

/** Default interface adapters exercised with portable contiguous arithmetic. */
@Suppress("LongParameterList") // BLAS operands and flags
private class PackedRegionBlas(private val delegate: DenseBlas) : DenseBlas by delegate {
    override fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) = regionGemv(this, alpha, a, x, beta, y, transpose, workspace)

    override fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        workspace: Workspace?,
    ) = regionGemm(this, alpha, a, transposeA, b, transposeB, beta, c, workspace)

    override fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) = regionSyrk(this, alpha, a, transpose, beta, c, lower, workspace)

    override fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) = regionTrsv(this, a, x, lower, transpose, unitDiag, workspace)

    override fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) = regionTrsm(this, a, b, lower, transpose, unitDiag, right, alpha, workspace)
}
