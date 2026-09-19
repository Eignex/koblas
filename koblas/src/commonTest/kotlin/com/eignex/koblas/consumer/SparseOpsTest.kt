package com.eignex.koblas.consumer

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.addScaled
import com.eignex.koblas.assertClose
import com.eignex.koblas.gemm
import com.eignex.koblas.gemmInto
import com.eignex.koblas.gemv
import com.eignex.koblas.gemvInto
import com.eignex.koblas.prepare
import com.eignex.koblas.sparse.ReferenceSparseBlas
import com.eignex.koblas.symm
import com.eignex.koblas.symv
import com.eignex.koblas.syrk
import com.eignex.koblas.transpose
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/** The sparse surface as a consumer reaches it: root-package extensions against the independent oracle. */
class SparseOpsTest {
    @Test
    fun `prepare is available from the root package`() {
        val source = matrix()
        val expected = DoubleArray(source.rows)
        ReferenceSparseBlas.gemv(1.0, source, INPUT, 0.0, expected)

        val prepared = source.prepare()
        source.values.fill(Double.NaN)
        val actual = DoubleArray(source.rows)
        prepared.gemv(1.0, INPUT, 0.0, actual)

        assertContentEquals(expected, actual)
    }

    @Test
    fun `the allocating and in-place matrix-vector extensions agree with the reference`() {
        val source = matrix()
        val expected = DoubleArray(source.rows)
        ReferenceSparseBlas.gemv(1.0, source, INPUT, 0.0, expected)
        val expectedInto = doubleArrayOf(2.0, -3.0)
        ReferenceSparseBlas.gemv(0.5, source, INPUT, -2.0, expectedInto, transpose = true)

        val allocated = source.gemv(INPUT)
        val actualInto = doubleArrayOf(2.0, -3.0)
        source.gemvInto(0.5, INPUT, -2.0, actualInto, transpose = true)

        assertContentEquals(expected, allocated)
        assertContentEquals(expectedInto, actualInto)
    }

    @Test
    fun `symmetric vector extensions agree with the reference`() {
        val source = matrix()
        val expected = DoubleArray(source.rows)
        ReferenceSparseBlas.symv(1.0, source, INPUT, 0.0, expected)

        val allocated = source.symv(INPUT)
        val actual = doubleArrayOf(2.0, -3.0)
        val expectedInto = actual.copyOf()
        ReferenceSparseBlas.symv(0.5, source, INPUT, -2.0, expectedInto)
        source.symv(0.5, INPUT, -2.0, actual)

        assertContentEquals(expected, allocated)
        assertContentEquals(expectedInto, actual)
    }

    @Test
    fun `symmetric matrix extension agrees with the reference`() {
        val source = matrix()
        val right = DenseMatrix.wrap(2, 2, doubleArrayOf(7.0, 11.0, -2.0, 5.0))
        val expected = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val actual = DenseMatrix.wrap(2, 2, expected.values.copyOf())
        ReferenceSparseBlas.symm(0.75, source, right, -0.5, expected)

        source.symm(0.75, right, -0.5, actual, workspace = Workspace())

        assertClose(expected, actual, "root symm")
    }

    @Test
    fun `the dense destination extensions agree with the reference on both operand kinds`() {
        val source = matrix()
        val dense = DenseMatrix.wrap(2, 2, doubleArrayOf(7.0, 11.0, -2.0, 5.0))
        val expectedDense = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val actualDense = DenseMatrix.wrap(2, 2, expectedDense.values.copyOf())
        ReferenceSparseBlas.gemm(0.75, source, false, dense, false, -0.5, expectedDense)
        source.gemmInto(0.75, false, dense, false, -0.5, actualDense, workspace = Workspace())
        assertClose(expectedDense, actualDense, "root sparse-dense gemmInto")

        val other = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to -1.0), listOf(1 to 4.0)))
        val expectedSparse = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val actualSparse = DenseMatrix.wrap(2, 2, expectedSparse.values.copyOf())
        ReferenceSparseBlas.gemm(0.75, source, false, other, true, -0.5, expectedSparse)
        source.gemmInto(0.75, false, other, true, -0.5, actualSparse, workspace = Workspace())
        assertClose(expectedSparse, actualSparse, "root sparse-sparse gemmInto")

        assertEquals(
            ReferenceSparseBlas.gemm(0.75, source, false, other, true),
            source.gemm(0.75, false, other, true),
            "root sparse-sparse gemm",
        )
    }

    @Test
    fun `rank product extensions agree with the reference`() {
        val source = matrix()
        val expectedSparse = ReferenceSparseBlas.syrk(source)
        val expectedDense = DenseMatrix.zero(source.rows)
        val actualDense = DenseMatrix.zero(source.rows)
        ReferenceSparseBlas.syrk(1.25, source, false, 0.0, expectedDense)

        val actualSparse = source.syrk()
        source.syrk(1.25, false, 0.0, actualDense, workspace = Workspace())

        assertEquals(expectedSparse, actualSparse)
        assertClose(expectedDense, actualDense, "root syrk")
    }

    @Test
    fun `scaled addition and transpose extensions agree with the reference`() {
        val source = matrix()
        val other = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to -1.0), listOf(1 to 4.0)))

        assertEquals(ReferenceSparseBlas.addScaled(-0.5, source, true, other), source.addScaled(-0.5, true, other))
        assertEquals(ReferenceSparseBlas.transpose(source), source.transpose())
    }

    private fun matrix(): SparseMatrix = SparseMatrix.ofColumns(
        2,
        2,
        listOf(listOf(0 to 2.0, 1 to 3.0), listOf(1 to 5.0)),
    )

    private companion object {
        val INPUT = doubleArrayOf(7.0, 11.0)
    }
}
