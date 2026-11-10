package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.storage.dense.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

class StagingTest {
    /**
     * A matrix-vector product whose input is also its destination, against the oracle's answer for the same
     * numbers held apart.
     */
    @Test
    fun `a staged vector gives the answer the reference computes from unaliased operands`() {
        val rng = Random(20260927)
        val n = 7
        val workspace = Scratch()
        for (transpose in booleanArrayOf(false, true)) {
            val a = randomMatrix(n, n, rng)
            val start = randomVector(n, rng)
            val expected = start.copyOf()
            ReferenceBlas.gemv(0.875, a, start.copyOf(), -0.25, expected, transpose)
            val aliased = start.copyOf()

            blas.kernels.gemv(0.875, a, aliased, 0, 1, -0.25, aliased, 0, 1, transpose, scratch = workspace)

            assertClose(expected, aliased, "gemv transpose=$transpose")
        }
    }

    /**
     * Every Level 2 routine that can snapshot an operand takes that snapshot from the workspace.
     *
     * Which overlap is reachable differs by routine, because a matrix's array is as long as all of it while
     * a vector is as long as one side: a product reaches the matrix through a single column, and a routine
     * requiring a square matrix reaches it only at order one. What is asserted is the loan either way, since
     * the answers these produce are the alias suite's subject.
     */
    @Test
    fun `every level two routine stages its alias from the workspace`() {
        val rng = Random(20260932)
        val n = 4

        assertStaged(n, "gemv over the vector it writes") { w ->
            val y = randomVector(n, rng)
            blas.kernels.gemv(0.875, randomMatrix(n, n, rng), y, 0, 1, -0.25, y, 0, 1, false, scratch = w)
        }
        assertStaged(n, "gemv over the matrix it writes") { w ->
            val y = randomVector(n, rng)
            blas.kernels.gemv(
                0.875, DenseMatrix.wrap(n, 1, y), doubleArrayOf(2.0), 0, 1, -0.25, y, 0, 1, false, scratch = w,
            )
        }
        assertStaged(n, "symv over the vector it writes") { w ->
            val y = randomVector(n, rng)
            blas.kernels.symv(0.875, randomMatrix(n, n, rng), y, 0, 1, -0.25, y, 0, 1, true, scratch = w)
        }
        assertStaged(n, "ger over the vector that is its matrix") { w ->
            val values = randomVector(n, rng)
            blas.kernels.ger(0.875, values, 0, 1, doubleArrayOf(2.0), 0, 1, DenseMatrix.wrap(n, 1, values), scratch = w)
        }
        assertStaged(1, "syr over the vector that is its matrix") { w ->
            val values = doubleArrayOf(2.0)
            blas.kernels.syr(0.875, values, 0, 1, one(values), true, scratch = w)
        }
        assertStaged(1, "syr2 over the vector that is its matrix") { w ->
            val values = doubleArrayOf(2.0)
            val other = DenseVector.wrap(doubleArrayOf(3.0))
            blas.kernels.syr2(
                0.875,
                values,
                0,
                1,
                other.values,
                other.offset,
                other.stride,
                one(values),
                true,
                scratch = w,
            )
        }
        assertStaged(1, "trsv over the triangle that is its right-hand side") { w ->
            val values = doubleArrayOf(2.0)
            blas.kernels.trsv(one(values), values, 0, 1, lower = true, scratch = w)
        }
        assertStaged(1, "trmv over the triangle that is its right-hand side") { w ->
            val values = doubleArrayOf(2.0)
            blas.kernels.trmv(one(values), values, 0, 1, lower = true, scratch = w)
        }
    }

    /** The one-entry matrix over [values], which is the only shape a square routine can alias through. */
    private fun one(values: DoubleArray): DenseMatrix = DenseMatrix.wrap(1, 1, values)

    /** That [call] took exactly one loan, of [length], and returned it. */
    private fun assertStaged(length: Int, what: String, call: (Scratch) -> Unit) {
        val workspace = Scratch()

        call(workspace)

        assertEquals(1, workspace.available(length), "$what did not stage from the workspace")
        assertEquals(1, workspace.idleLengths(), "$what borrowed scratch besides its snapshot")
    }

    private val blas: KoblasEngine get() = KoblasEngine.scalar

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
        val a = DenseMatrix.wrap(3, 1, actual, leadingDimension = 5)
        val expected = actual.copyOf()
        ReferenceBlas.gemv(0.875, contiguousCopy(a), doubleArrayOf(2.0), -0.25, expected)

        blas.kernels.gemv(0.875, a, doubleArrayOf(2.0), 0, 1, -0.25, actual, 0, 1, scratch = Scratch())

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

    @Test
    fun `a staged window holds the entries it addresses and no more`() {
        val original = DoubleArray(64) { it.toDouble() }
        val backing = original.copyOf()
        for ((offset, count, stride) in listOf(
            Triple(0, 64, 1),
            Triple(40, 4, 1),
            Triple(9, 4, 5),
            Triple(33, 4, -8),
        )) {
            val expected = DoubleArray(count) { backing[offset + it * stride] }
            val scratch = Scratch()

            stagedWindow(
                scratch,
                DenseVector.wrap(backing, offset, count, stride),
                aliased = true,
            ) { copy ->
                assertEquals(count, copy.size, "$offset/$count/$stride staged its buffer rather than its entries")
                assertContentEquals(expected, copy.toDoubleArray())
            }

            assertEquals(1, scratch.available(count), "$offset/$count/$stride did not hand its loan back")
            assertContentEquals(original, backing, "$offset/$count/$stride was written through")
        }
    }

    @Test
    fun `a disjoint window is passed through without scratch`() {
        val backing = DoubleArray(64)
        val scratch = Scratch()

        val view = DenseVector.wrap(backing, 33, 4, -8)
        stagedWindow(scratch, view, aliased = false) { stable ->
            assertSame(view, stable)
        }

        assertEquals(0, scratch.idleLengths())
    }

    @Test
    fun `identical mappings share one snapshot`() {
        for ((offset, count, stride) in listOf(Triple(0, 4, 1), Triple(7, 4, -2), Triple(2, 1, 3))) {
            val backing = DoubleArray(8) { it.toDouble() }
            val destination = DenseMatrix.wrap(4, 2, backing)
            val x = DenseVector.wrap(backing, offset, count, stride)
            val y = DenseVector.wrap(backing, offset, count, if (count == 1) 1 else stride)
            val expected = x.toDoubleArray()
            val scratch = Scratch()

            stagedWindows(scratch, x, y, destination) { sx, sy ->
                backing.fill(-1.0)
                assertSame(sx, sy)
                assertContentEquals(expected, sx.toDoubleArray())
            }

            assertEquals(1, scratch.available(count))
        }
    }

    @Test
    fun `aliased vector calls reuse their scratch`() {
        for (symmetric in booleanArrayOf(false, true)) {
            val a = DenseMatrix.wrap(3, 3, DoubleArray(9) { 1.0 + it })
            val workspace = Scratch()
            val actual = DenseVector.wrap(doubleArrayOf(1.0, 2.0, 3.0))
            val expected = actual.toDoubleArray()

            repeat(3) {
                val input = expected.copyOf()
                if (symmetric) {
                    ReferenceBlas.symv(0.5, a, input, 0.25, expected)
                    KoblasEngine.scalar.symv(0.5, a, actual, 0.25, actual, scratch = workspace)
                } else {
                    ReferenceBlas.gemv(0.5, a, input, 0.25, expected)
                    KoblasEngine.scalar.gemv(0.5, a, actual, 0.25, actual, scratch = workspace)
                }
            }

            assertClose(expected, actual.toDoubleArray(), "aliased matrix vector symmetric=$symmetric")
            assertEquals(1, workspace.available(actual.size))
            assertEquals(1, workspace.idleLengths())
        }
    }

    @Test
    fun `windows overlap only where they share an addressed entry`() {
        val backing = DoubleArray(32)
        val other = DoubleArray(32)

        assertTrue(windowsOverlap(backing, 0, 8, 2, backing, 4, 4, 2))
        assertFalse(windowsOverlap(backing, 0, 8, 2, backing, 1, 8, 2))
        assertTrue(windowsOverlap(backing, 31, 4, -3, backing, 22, 2, 1))
        assertFalse(windowsOverlap(backing, 0, 8, 1, other, 0, 8, 1))
    }
}
