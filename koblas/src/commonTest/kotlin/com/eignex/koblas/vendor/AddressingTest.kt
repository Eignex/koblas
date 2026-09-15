package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow
import kotlin.test.Test
import kotlin.test.assertEquals

class AddressingTest {
    private val storage = DoubleArray(400) { it.toDouble() }

    @Test
    fun `a column major panel keeps its leading dimension`() {
        val window = MatrixWindow(storage, 4, 3, offset = 7, rowStride = 1, columnStride = 10)

        val addressing = addressingOf(window)

        assertEquals(Addressing.ColumnMajor, addressing)
        assertEquals(10, leadingDimension(window, addressing))
    }

    @Test
    fun `a transposed column major panel is row major and still needs no copy`() {
        val window = MatrixWindow(storage, 4, 3, offset = 7, rowStride = 1, columnStride = 10).transpose()

        val addressing = addressingOf(window)

        assertEquals(Addressing.RowMajor, addressing)
        assertEquals(10, leadingDimension(window, addressing))
    }

    @Test
    fun `a window with no unit stride is staged`() {
        val window = MatrixWindow(storage, 3, 3, offset = 0, rowStride = 2, columnStride = 20)

        assertEquals(Addressing.Staged, addressingOf(window))
    }

    @Test
    fun `a negative column stride is staged because blas has no negative leading dimension`() {
        val window = MatrixWindow(storage, 3, 3, offset = 300, rowStride = 1, columnStride = -20)

        assertEquals(Addressing.Staged, addressingOf(window))
    }

    @Test
    fun `a structured block off the parent diagonal is staged`() {
        val parent = MatrixWindow(storage, 8, 8, structure = MatrixStructure.SymmetricLower)

        val onDiagonal = parent.window(2, 3, 2, 3)
        val offDiagonal = parent.window(4, 3, 1, 3)

        assertEquals(Addressing.ColumnMajor, addressingOf(onDiagonal))
        assertEquals(Addressing.Staged, addressingOf(offDiagonal))
    }

    @Test
    fun `a call runs row major only when every matrix already is`() {
        val columnMajor = MatrixWindow(storage, 4, 4)
        val rowMajor = MatrixWindow(storage, 4, 4).transpose()

        assertEquals(Cblas.ROW_MAJOR, layoutOf(listOf(rowMajor, rowMajor)))
        assertEquals(Cblas.COL_MAJOR, layoutOf(listOf(rowMajor, columnMajor)))
        assertEquals(Cblas.COL_MAJOR, layoutOf(listOf(columnMajor, columnMajor)))
        assertEquals(Cblas.COL_MAJOR, layoutOf(emptyList()))
    }

    @Test
    fun `a transposed gemm operand reconciles with a flag instead of a copy`() {
        val a = MatrixWindow(storage, 4, 4).transpose()
        val b = MatrixWindow(storage, 4, 4)
        val c = MatrixWindow(storage, 4, 4)

        val addressing = effectiveAddressing(VendorOperation.Gemm, listOf(a, b, c))

        assertEquals(listOf(Addressing.RowMajor, Addressing.ColumnMajor, Addressing.ColumnMajor), addressing)
        assertEquals(Cblas.TRANS, transposeFor(addressing[0], Cblas.COL_MAJOR))
        assertEquals(Cblas.NO_TRANS, transposeFor(addressing[1], Cblas.COL_MAJOR))
    }

    @Test
    fun `the symm operand with no flag of its own is copied when it disagrees`() {
        val a = MatrixWindow(storage, 4, 4, structure = MatrixStructure.SymmetricLower)
        val b = MatrixWindow(storage, 4, 4).transpose()
        val c = MatrixWindow(storage, 4, 4)

        val addressing = effectiveAddressing(VendorOperation.Symm, listOf(a, b, c))

        assertEquals(Addressing.Staged, addressing[1])
    }

    @Test
    fun `the shared syr2k flag copies the second operand only when the two disagree`() {
        val columnMajor = MatrixWindow(storage, 4, 4)
        val rowMajor = MatrixWindow(storage, 4, 4).transpose()
        val c = MatrixWindow(storage, 4, 4, structure = MatrixStructure.SymmetricLower)

        val agreeing = effectiveAddressing(VendorOperation.Syr2k, listOf(columnMajor, columnMajor, c))
        val disagreeing = effectiveAddressing(VendorOperation.Syr2k, listOf(columnMajor, rowMajor, c))

        assertEquals(Addressing.ColumnMajor, agreeing[1])
        assertEquals(Addressing.Staged, disagreeing[1])
    }

    @Test
    fun `reading a stored triangle under the opposite layout flips which triangle blas finds`() {
        val lower = MatrixWindow(storage, 4, 4, structure = MatrixStructure.SymmetricLower)

        assertEquals(Cblas.LOWER, uploFor(lower, Addressing.ColumnMajor, Cblas.COL_MAJOR))
        assertEquals(Cblas.UPPER, uploFor(lower, Addressing.ColumnMajor, Cblas.ROW_MAJOR))
    }

    @Test
    fun `an implicit unit diagonal is declared rather than read`() {
        val unit = MatrixWindow(storage, 4, 4, structure = MatrixStructure.UnitLower)
        val stored = MatrixWindow(storage, 4, 4, structure = MatrixStructure.TriangularLower)

        assertEquals(Cblas.UNIT, diagFor(unit))
        assertEquals(Cblas.NON_UNIT, diagFor(stored))
    }

    @Test
    fun `a negatively strided vector passes its low end with the sign kept`() {
        val window = VectorWindow(storage, size = 5, offset = 40, stride = -10)

        assertEquals(0, baseIndex(window))
        assertEquals(40, baseIndex(VectorWindow(storage, size = 5, offset = 40, stride = 10)))
    }
}
