package com.eignex.koblas.consumer

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.addScaled
import com.eignex.koblas.assertClose
import com.eignex.koblas.prepare
import com.eignex.koblas.sparse.ReferenceSparseBlas
import com.eignex.koblas.symm
import com.eignex.koblas.symv
import com.eignex.koblas.syrk
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

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
        val actual = DenseMatrix.wrap(2, 2, expected.data.copyOf())
        ReferenceSparseBlas.symm(0.75, source, right, -0.5, expected, workspace = Workspace())

        source.symm(0.75, right, -0.5, actual, workspace = Workspace())

        assertClose(expected, actual, "root symm")
    }

    @Test
    fun `rank product extensions agree with the reference`() {
        val source = matrix()
        val expectedSparse = ReferenceSparseBlas.syrk(source)
        val expectedDense = DenseMatrix.zero(source.rows)
        val actualDense = DenseMatrix.zero(source.rows)
        ReferenceSparseBlas.syrk(1.25, source, false, 0.0, expectedDense, workspace = Workspace())

        val actualSparse = source.syrk()
        source.syrk(1.25, false, 0.0, actualDense, workspace = Workspace())

        assertEquals(expectedSparse, actualSparse)
        assertClose(expectedDense, actualDense, "root syrk")
    }

    @Test
    fun `scaled addition extension agrees with the reference`() {
        val source = matrix()
        val other = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to -1.0), listOf(1 to 4.0)))
        val expected = ReferenceSparseBlas.addScaled(-0.5, source, true, other)

        val actual = source.addScaled(-0.5, true, other)

        assertEquals(expected, actual)
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
